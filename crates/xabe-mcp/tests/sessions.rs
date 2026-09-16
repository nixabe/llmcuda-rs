use serde_json::{Value, json};
use std::{path::PathBuf, time::Duration};
use xabe_mcp::{CancelToken, Config, Error, Registry};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("xabe-mcp-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn config(&self) -> Config {
        serde_json::from_value(json!({"timeout_ms":2000,"mcpServers":{
            "test":{"command":"python3", "args":["-u",format!("{}/tests/fixtures/server.py",env!("CARGO_MANIFEST_DIR")),self.0]}
        }})).unwrap()
    }
    fn events(&self) -> Vec<Value> {
        std::fs::read_dir(&self.0)
            .unwrap()
            .flat_map(|entry| {
                std::fs::read_to_string(entry.unwrap().path())
                    .unwrap()
                    .lines()
                    .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tool(session: &xabe_mcp::Session, name: &str) -> String {
    session
        .tools()
        .iter()
        .find(|t| t.original_name == name)
        .unwrap()
        .name
        .clone()
}

#[tokio::test]
async fn stdio_discovery_is_paginated_and_sessions_have_separate_processes() {
    let fixture = Fixture::new();
    let registry = Registry::new(fixture.config()).unwrap();
    let a = registry.create(&["test".into()]).await.unwrap();
    let b = registry.create(&["test".into()]).await.unwrap();
    assert_eq!(a.tools().len(), 4);
    let input = json!({"text":"你好\n  spaces", "nested":[true,42,{"x":"y"}]})
        .as_object()
        .unwrap()
        .clone();
    let ra = a
        .call(&tool(&a, "echo"), input.clone(), &CancelToken::new())
        .await
        .unwrap();
    let rb = b
        .call(&tool(&b, "echo"), input.clone(), &CancelToken::new())
        .await
        .unwrap();
    assert_eq!(
        ra.structured_content.as_ref().unwrap()["arguments"],
        json!(input)
    );
    assert_ne!(
        ra.structured_content.unwrap()["pid"],
        rb.structured_content.unwrap()["pid"]
    );
    registry.remove(&a.id).unwrap();
    assert!(registry.get(&a.id).is_err());
    assert!(matches!(
        a.call(&tool(&a, "echo"), Default::default(), &CancelToken::new())
            .await,
        Err(Error::Cancelled)
    ));
    registry.remove(&b.id).unwrap();
}

#[tokio::test]
async fn filtering_capacity_leases_and_ttl_are_enforced() {
    let fixture = Fixture::new();
    let mut config = fixture.config();
    config.max_sessions = 1;
    config.session_ttl_ms = 150;
    config.servers.get_mut("test").unwrap().allowed_tools = Some(vec!["echo".into()]);
    let registry = Registry::new(config).unwrap();
    assert!(registry.create(&["missing".into()]).await.is_err());
    let a = registry.create(&["test".into()]).await.unwrap();
    assert_eq!(a.tools().len(), 1);
    assert!(matches!(
        registry.create(&["test".into()]).await,
        Err(Error::Capacity)
    ));
    let lease = a.acquire().unwrap();
    assert!(a.acquire().is_err());
    drop(lease);
    assert!(a.acquire().is_ok());
    assert!(matches!(
        a.call("not-offered", Default::default(), &CancelToken::new())
            .await,
        Err(Error::Tool(_))
    ));
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(registry.get(&a.id).is_err());
    assert!(a.cancellation().is_cancelled());
}

#[tokio::test]
async fn calls_timeout_once_and_cancellation_is_forwarded() {
    let fixture = Fixture::new();
    let mut config = fixture.config();
    config.timeout_ms = 200;
    let registry = Registry::new(config).unwrap();
    let session = registry.create(&["test".into()]).await.unwrap();
    assert!(matches!(
        session
            .call(
                &tool(&session, "slow"),
                Default::default(),
                &CancelToken::new()
            )
            .await,
        Err(Error::Timeout)
    ));
    tokio::time::sleep(Duration::from_millis(100)).await;
    let events = fixture.events();
    assert_eq!(
        events
            .iter()
            .filter(|e| e["method"] == "tools/call")
            .count(),
        1
    );
    assert!(
        events
            .iter()
            .any(|e| e["method"] == "notifications/cancelled")
    );
    // Cancelling before a call never sends it.
    let cancel = CancelToken::new();
    cancel.cancel();
    assert!(matches!(
        session
            .call(&tool(&session, "echo"), Default::default(), &cancel)
            .await,
        Err(Error::Cancelled)
    ));
    registry.remove(&session.id).unwrap();
}

#[tokio::test]
async fn tool_errors_are_results_and_oversized_outputs_are_refused() {
    let fixture = Fixture::new();
    let mut config = fixture.config();
    config.max_result_bytes = 4096;
    let registry = Registry::new(config).unwrap();
    let session = registry.create(&["test".into()]).await.unwrap();
    let result = session
        .call(
            &tool(&session, "error"),
            Default::default(),
            &CancelToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.is_error, Some(true));
    assert!(matches!(
        session
            .call(
                &tool(&session, "huge"),
                Default::default(),
                &CancelToken::new()
            )
            .await,
        Err(Error::Size("result"))
    ));
    registry.remove(&session.id).unwrap();
}

#[tokio::test]
async fn streamable_http_discovers_and_executes_with_session_header() {
    use axum::{
        Json, Router,
        http::{HeaderMap, StatusCode},
        response::IntoResponse,
        routing::post,
    };
    async fn rpc(
        axum::extract::State(counts): axum::extract::State<
            std::sync::Arc<(
                std::sync::atomic::AtomicUsize,
                std::sync::atomic::AtomicUsize,
            )>,
        >,
        headers: HeaderMap,
        Json(request): Json<Value>,
    ) -> axum::response::Response {
        if request.get("id").is_none() {
            return StatusCode::ACCEPTED.into_response();
        }
        let method = request["method"].as_str().unwrap();
        if method == "initialize" {
            counts.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        if method == "tools/call" {
            counts.1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if request["params"]["arguments"]["expired"] == true {
                return StatusCode::NOT_FOUND.into_response();
            }
        }
        let result = match method {
            "initialize" => {
                json!({"protocolVersion":request["params"]["protocolVersion"],"capabilities":{"tools":{}},"serverInfo":{"name":"http-fixture","version":"1"}})
            }
            "tools/list" => json!({"tools":[{"name":"echo","inputSchema":{"type":"object"}}]}),
            "tools/call" => {
                assert_eq!(headers.get("mcp-session-id").unwrap(), "fixture-session");
                json!({"content":[{"type":"text","text":"http ok"}],"structuredContent":request["params"]["arguments"]})
            }
            _ => json!({}),
        };
        let wire = json!({"jsonrpc":"2.0","id":request["id"],"result":result});
        if method == "tools/call" && request["params"]["arguments"]["sse"] == true {
            return (
                [("content-type", "text/event-stream")],
                format!("event: message\ndata: {wire}\n\n"),
            )
                .into_response();
        }
        let mut response = Json(wire).into_response();
        if method == "initialize" {
            response
                .headers_mut()
                .insert("mcp-session-id", "fixture-session".parse().unwrap());
        }
        response
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let counts = std::sync::Arc::new((
        std::sync::atomic::AtomicUsize::new(0),
        std::sync::atomic::AtomicUsize::new(0),
    ));
    let server_counts = counts.clone();
    let task = tokio::spawn(async move {
        axum::serve(
            listener,
            Router::new()
                .route(
                    "/mcp",
                    post(rpc).delete(|| async { StatusCode::NO_CONTENT }),
                )
                .with_state(server_counts),
        )
        .await
        .unwrap();
    });
    let registry = Registry::new(
        serde_json::from_value(
            json!({"mcpServers":{"http":{"url":format!("http://{address}/mcp")}}}),
        )
        .unwrap(),
    )
    .unwrap();
    let session = registry.create(&["http".into()]).await.unwrap();
    let result = session
        .call(
            &tool(&session, "echo"),
            json!({"a":1}).as_object().unwrap().clone(),
            &CancelToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(result.structured_content, Some(json!({"a":1})));
    let sse = session
        .call(
            &tool(&session, "echo"),
            json!({"sse":true}).as_object().unwrap().clone(),
            &CancelToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(sse.structured_content, Some(json!({"sse":true})));
    assert!(
        session
            .call(
                &tool(&session, "echo"),
                json!({"expired":true}).as_object().unwrap().clone(),
                &CancelToken::new()
            )
            .await
            .is_err()
    );
    assert_eq!(
        counts.0.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "must not reinitialize and replay on HTTP 404"
    );
    assert_eq!(
        counts.1.load(std::sync::atomic::Ordering::SeqCst),
        3,
        "each tool call is sent exactly once"
    );
    registry.remove(&session.id).unwrap();
    task.abort();
}

#[tokio::test]
async fn cancelling_an_active_call_sends_a_notification_without_replay() {
    let fixture = Fixture::new();
    let registry = Registry::new(fixture.config()).unwrap();
    let session = registry.create(&["test".into()]).await.unwrap();
    let cancel = CancelToken::new();
    let task_session = session.clone();
    let task_cancel = cancel.clone();
    let task = tokio::spawn(async move {
        task_session
            .call(
                &tool(&task_session, "slow"),
                Default::default(),
                &task_cancel,
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while !fixture.events().iter().any(|e| e["method"] == "tools/call") {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    cancel.cancel();
    assert!(matches!(task.await.unwrap(), Err(Error::Cancelled)));
    tokio::time::timeout(Duration::from_secs(1), async {
        while !fixture
            .events()
            .iter()
            .any(|e| e["method"] == "notifications/cancelled")
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        fixture
            .events()
            .iter()
            .filter(|e| e["method"] == "tools/call")
            .count(),
        1
    );
    registry.remove(&session.id).unwrap();
}

#[test]
fn configuration_rejects_ambiguous_transports_and_unbounded_limits() {
    for config in [
        json!({}),
        json!({"max_sessions":0,"mcpServers":{"x":{"command":"test"}}}),
        json!({"mcpServers":{"x":{"command":"test","url":"http://localhost"}}}),
        json!({"mcpServers":{"x":{"url":"file:///etc/passwd"}}}),
        json!({"mcpServers":{"bad.name":{"command":"test"}}}),
    ] {
        let config: Config = serde_json::from_value(config).unwrap();
        assert!(Registry::new(config).is_err());
    }
}
