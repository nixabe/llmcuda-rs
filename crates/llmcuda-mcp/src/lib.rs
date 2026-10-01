//! MCP clients live above inference: one connection set per explicit session.
//! Server definitions are administrator configuration, never request-supplied
//! commands or URLs. Tool calls are sent once and are never replayed on failure.

use rmcp::{
    Peer, RoleClient, ServiceExt,
    model::{
        CallToolRequest, CallToolRequestParams, CancelledNotificationParam, ClientRequest,
        RequestId, ServerResult,
    },
    service::{PeerRequestOptions, RunningService},
    transport::{
        StreamableHttpClientTransport, TokioChildProcess,
        streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

pub use rmcp::model::CallToolResult;
pub use tokio_util::sync::CancellationToken as CancelToken;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid MCP configuration: {0}")]
    Config(String),
    #[error("unknown or expired MCP session")]
    Session,
    #[error("unknown MCP tool: {0}")]
    Tool(String),
    #[error("MCP capacity exceeded")]
    Capacity,
    #[error("MCP operation cancelled")]
    Cancelled,
    #[error("MCP operation timed out; execution outcome may be unknown")]
    Timeout,
    #[error("MCP {0} exceeds the configured size limit")]
    Size(&'static str),
    #[error("MCP protocol or transport failure: {0}")]
    Transport(String),
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub cwd: Option<String>,
    pub url: Option<String>,
    pub bearer_token_env: Option<String>,
    /// Original MCP names. None offers all; an empty list offers none.
    pub allowed_tools: Option<Vec<String>>,
}

#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    #[serde(rename = "mcpServers")]
    pub servers: BTreeMap<String, ServerConfig>,
    pub max_sessions: usize,
    pub max_concurrent_calls: usize,
    pub max_tools: usize,
    pub timeout_ms: u64,
    pub session_ttl_ms: u64,
    pub max_result_bytes: usize,
    pub max_argument_bytes: usize,
    pub max_iterations: u32,
    pub max_agent_tokens: u32,
    pub max_agent_ms: u64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            servers: BTreeMap::new(),
            max_sessions: 32,
            max_concurrent_calls: 8,
            max_tools: 256,
            timeout_ms: 30_000,
            session_ttl_ms: 1_800_000,
            max_result_bytes: 1024 * 1024,
            max_argument_bytes: 256 * 1024,
            max_iterations: 8,
            max_agent_tokens: 8192,
            max_agent_ms: 300_000,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<(), Error> {
        if self.servers.is_empty()
            || self.max_sessions == 0
            || self.max_sessions > 1024
            || self.max_concurrent_calls == 0
            || self.max_concurrent_calls > 1024
            || self.max_tools == 0
            || self.max_tools > 4096
            || self.timeout_ms == 0
            || self.timeout_ms > 3_600_000
            || self.session_ttl_ms == 0
            || self.session_ttl_ms > 86_400_000
            || self.max_result_bytes == 0
            || self.max_result_bytes > 64 * 1024 * 1024
            || self.max_argument_bytes == 0
            || self.max_argument_bytes > 1024 * 1024
            || self.max_iterations == 0
            || self.max_iterations > 128
            || self.max_agent_tokens == 0
            || self.max_agent_ms == 0
            || self.max_agent_ms > 3_600_000
        {
            return Err(Error::Config(
                "empty servers or limits outside supported bounds".into(),
            ));
        }
        for (label, server) in &self.servers {
            if label.is_empty()
                || label.len() > 128
                || !label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            {
                return Err(Error::Config(
                    "server labels must use letters, digits, '_' or '-'".into(),
                ));
            }
            match (&server.command, &server.url) {
                (Some(command), None)
                    if !command.is_empty() && server.bearer_token_env.is_none() => {}
                (None, Some(url))
                    if (url.starts_with("https://") || url.starts_with("http://"))
                        && server.args.is_empty()
                        && server.env.is_empty()
                        && server.cwd.is_none() => {}
                _ => {
                    return Err(Error::Config(format!(
                        "{label}: choose either stdio command or HTTP URL"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RegisteredTool {
    pub name: String,
    pub server: String,
    pub original_name: String,
    pub description: String,
    pub input_schema: Value,
}
impl RegisteredTool {
    pub fn function_definition(&self) -> Value {
        json!({"type":"function", "name":self.name, "description":format!("{} / {}: {}", self.server, self.original_name, self.description), "parameters":self.input_schema})
    }
}

struct Connection {
    service: RunningService<RoleClient, ()>,
}
pub struct Session {
    pub id: String,
    tools: BTreeMap<String, RegisteredTool>,
    connections: BTreeMap<String, Connection>,
    cancel: CancellationToken,
    calls: Arc<Semaphore>,
    /// One conversation/tool operation at a time per stateful session.
    lease: Arc<Semaphore>,
    config: Arc<Config>,
    _permit: OwnedSemaphorePermit,
}
impl Session {
    pub fn tools(&self) -> Vec<RegisteredTool> {
        self.tools.values().cloned().collect()
    }
    pub fn cancellation(&self) -> CancellationToken {
        self.cancel.clone()
    }
    pub fn acquire(&self) -> Result<OwnedSemaphorePermit, Error> {
        if self.cancel.is_cancelled() {
            return Err(Error::Cancelled);
        }
        self.lease
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Capacity)
    }
    pub fn close(&self) {
        self.cancel.cancel();
        for connection in self.connections.values() {
            connection.service.cancellation_token().cancel();
        }
    }
    /// The caller holds `acquire()` for its complete operation or agent loop.
    pub async fn call(
        &self,
        name: &str,
        arguments: Map<String, Value>,
        cancel: &CancellationToken,
    ) -> Result<CallToolResult, Error> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| Error::Tool(name.into()))?;
        check_size(&arguments, self.config.max_argument_bytes, "arguments")?;
        let service = &self.connections[&tool.server].service;
        let operation = async {
            let _permit = self.calls.acquire().await.map_err(|_| Error::Cancelled)?;
            let params =
                CallToolRequestParams::new(tool.original_name.clone()).with_arguments(arguments);
            let handle = service
                .send_cancellable_request(
                    ClientRequest::CallToolRequest(CallToolRequest::new(params)),
                    PeerRequestOptions::no_options(),
                )
                .await
                .map_err(|e| Error::Transport(e.to_string()))?;
            let mut guard = CancelRequest {
                peer: service.peer().clone(),
                id: Some(handle.id.clone()),
            };
            let response = handle
                .await_response()
                .await
                .map_err(|e| Error::Transport(e.to_string()))?;
            guard.id = None;
            let ServerResult::CallToolResult(result) = response else {
                return Err(Error::Transport(
                    "tool requires an unsupported input/task round".into(),
                ));
            };
            check_size(&result, self.config.max_result_bytes, "result")?;
            Ok(result)
        };
        tokio::select! {
            biased;
            _ = self.cancel.cancelled() => Err(Error::Cancelled),
            _ = cancel.cancelled() => Err(Error::Cancelled),
            result = tokio::time::timeout(Duration::from_millis(self.config.timeout_ms), operation) => result.map_err(|_| Error::Timeout)?,
        }
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        self.close();
    }
}

struct CancelRequest {
    peer: Peer<RoleClient>,
    id: Option<RequestId>,
}
impl Drop for CancelRequest {
    fn drop(&mut self) {
        if let Some(request_id) = self.id.take() {
            let peer = self.peer.clone();
            tokio::spawn(async move {
                let _ = tokio::time::timeout(
                    Duration::from_secs(1),
                    peer.notify_cancelled(CancelledNotificationParam::new(
                        Some(request_id),
                        Some("caller cancelled or timed out".into()),
                    )),
                )
                .await;
            });
        }
    }
}

pub struct Registry {
    config: Arc<Config>,
    sessions: Mutex<HashMap<String, Arc<Session>>>,
    slots: Arc<Semaphore>,
    calls: Arc<Semaphore>,
}
impl Registry {
    pub fn new(config: Config) -> Result<Arc<Self>, Error> {
        config.validate()?;
        Ok(Arc::new(Self {
            slots: Arc::new(Semaphore::new(config.max_sessions)),
            calls: Arc::new(Semaphore::new(config.max_concurrent_calls)),
            config: Arc::new(config),
            sessions: Mutex::new(HashMap::new()),
        }))
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn get(&self, id: &str) -> Result<Arc<Session>, Error> {
        self.sessions
            .lock()
            .expect("MCP sessions mutex")
            .get(id)
            .filter(|s| !s.cancel.is_cancelled())
            .cloned()
            .ok_or(Error::Session)
    }
    pub fn remove(&self, id: &str) -> Result<(), Error> {
        let session = self
            .sessions
            .lock()
            .expect("MCP sessions mutex")
            .remove(id)
            .ok_or(Error::Session)?;
        session.close();
        Ok(())
    }
    pub async fn create(self: &Arc<Self>, labels: &[String]) -> Result<Arc<Session>, Error> {
        if labels.is_empty() {
            return Err(Error::Config(
                "select at least one configured server".into(),
            ));
        }
        let mut seen = HashSet::new();
        for label in labels {
            if !self.config.servers.contains_key(label) || !seen.insert(label) {
                return Err(Error::Config(format!(
                    "unknown or duplicate server label: {label}"
                )));
            }
        }
        let permit = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Capacity)?;
        let connect = async {
            let mut connections = BTreeMap::new();
            let mut tools = BTreeMap::new();
            for label in labels {
                let config = &self.config.servers[label];
                let service = if let Some(command) = &config.command {
                    let mut cmd = tokio::process::Command::new(command);
                    cmd.args(&config.args).envs(&config.env).kill_on_drop(true);
                    if let Some(cwd) = &config.cwd {
                        cmd.current_dir(cwd);
                    }
                    let transport =
                        TokioChildProcess::new(cmd).map_err(|e| Error::Transport(e.to_string()))?;
                    ().serve(transport)
                        .await
                        .map_err(|e| Error::Transport(e.to_string()))?
                } else {
                    let mut transport = StreamableHttpClientTransportConfig::with_uri(
                        config.url.clone().expect("validated URL"),
                    )
                    .reinit_on_expired_session(false)
                    .max_sse_event_size(self.config.max_result_bytes);
                    if let Some(env) = &config.bearer_token_env {
                        transport = transport.auth_header(std::env::var(env).map_err(|_| {
                            Error::Config(format!("missing credential environment variable {env}"))
                        })?);
                    }
                    ().serve(StreamableHttpClientTransport::from_config(transport))
                        .await
                        .map_err(|e| Error::Transport(e.to_string()))?
                };
                // Bound paginated discovery and reject repeated cursors.
                let mut cursor = None;
                let mut cursors = HashSet::new();
                loop {
                    let page = service
                        .list_tools(cursor.clone().map(|cursor| {
                            rmcp::model::PaginatedRequestParams::default().with_cursor(Some(cursor))
                        }))
                        .await
                        .map_err(|e| Error::Transport(e.to_string()))?;
                    check_size(&page, self.config.max_result_bytes, "tool catalog")?;
                    for tool in page.tools {
                        if config
                            .allowed_tools
                            .as_ref()
                            .is_some_and(|allowed| !allowed.iter().any(|n| n == tool.name.as_ref()))
                        {
                            continue;
                        }
                        if tools.len() >= self.config.max_tools {
                            return Err(Error::Size("tool catalog"));
                        }
                        // Opaque aliases avoid delimiter ambiguity, unsupported name
                        // characters, and collisions between server/tool name pairs.
                        let name = format!("mcp_{}", tools.len());
                        let registered = RegisteredTool {
                            name: name.clone(),
                            server: label.clone(),
                            original_name: tool.name.to_string(),
                            description: tool
                                .description
                                .map(|d| d.to_string())
                                .unwrap_or_default(),
                            input_schema: Value::Object((*tool.input_schema).clone()),
                        };
                        if tools.values().any(|old: &RegisteredTool| {
                            old.server == registered.server
                                && old.original_name == registered.original_name
                        }) {
                            return Err(Error::Config("duplicate tool in catalog".into()));
                        }
                        tools.insert(name, registered);
                    }
                    cursor = page.next_cursor;
                    match &cursor {
                        None => break,
                        Some(cursor)
                            if cursors.len() < self.config.max_tools
                                && cursors.insert(cursor.clone()) => {}
                        _ => return Err(Error::Size("tool pagination")),
                    }
                }
                connections.insert(label.clone(), Connection { service });
            }
            check_size(&tools, self.config.max_result_bytes, "tool catalog")?;
            Ok::<_, Error>((connections, tools))
        };
        let (connections, tools) =
            tokio::time::timeout(Duration::from_millis(self.config.timeout_ms), connect)
                .await
                .map_err(|_| Error::Timeout)??;
        let session = Arc::new(Session {
            id: uuid::Uuid::new_v4().to_string(),
            tools,
            connections,
            cancel: CancellationToken::new(),
            calls: self.calls.clone(),
            lease: Arc::new(Semaphore::new(1)),
            config: self.config.clone(),
            _permit: permit,
        });
        self.sessions
            .lock()
            .expect("MCP sessions mutex")
            .insert(session.id.clone(), session.clone());
        let weak = Arc::downgrade(self);
        let id = session.id.clone();
        let ttl = self.config.session_ttl_ms;
        let cancel = session.cancel.clone();
        tokio::spawn(async move {
            tokio::select! { _ = cancel.cancelled() => {}, _ = tokio::time::sleep(Duration::from_millis(ttl)) => {
                if let Some(registry) = weak.upgrade() { let _ = registry.remove(&id); }
            }}
        });
        Ok(session)
    }
}
impl Drop for Registry {
    fn drop(&mut self) {
        for session in self
            .sessions
            .get_mut()
            .expect("MCP sessions mutex")
            .values()
        {
            session.close();
        }
    }
}

/// Bound serialization without first allocating an unbounded JSON string.
pub fn json_size(value: &impl Serialize, limit: usize, what: &'static str) -> Result<usize, Error> {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_sub(bytes.len())
                .ok_or_else(|| std::io::Error::other("size limit"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(limit);
    serde_json::to_writer(&mut counter, value).map_err(|_| Error::Size(what))?;
    Ok(limit - counter.0)
}

pub fn check_size(value: &impl Serialize, limit: usize, what: &'static str) -> Result<(), Error> {
    json_size(value, limit, what).map(|_| ())
}
