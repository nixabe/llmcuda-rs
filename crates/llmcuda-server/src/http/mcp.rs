//! Explicit MCP session and tool endpoints, behind the normal API-key middleware.
use super::{
    AppState,
    error::{ApiError, Dialect, parse_body},
};
use axum::{
    Json,
    body::Bytes,
    extract::{FromRef, Path, Query, State},
    http::StatusCode,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::sync::Arc;
const DIALECT: Dialect = Dialect::OpenAi;
pub(super) fn registry(state: &AppState) -> Result<&Arc<llmcuda_mcp::Registry>, ApiError> {
    state.mcp.as_ref().ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            DIALECT,
            "MCP is disabled; configure --mcp-servers-config",
        )
    })
}
pub(super) fn failure(error: llmcuda_mcp::Error) -> ApiError {
    let status = match &error {
        llmcuda_mcp::Error::Session | llmcuda_mcp::Error::Tool(_) => StatusCode::NOT_FOUND,
        llmcuda_mcp::Error::Config(_) | llmcuda_mcp::Error::Size(_) => StatusCode::BAD_REQUEST,
        llmcuda_mcp::Error::Capacity => StatusCode::CONFLICT,
        llmcuda_mcp::Error::Cancelled => StatusCode::CONFLICT,
        llmcuda_mcp::Error::Timeout => StatusCode::GATEWAY_TIMEOUT,
        llmcuda_mcp::Error::Transport(_) => StatusCode::BAD_GATEWAY,
    };
    ApiError::new(status, DIALECT, error.to_string())
}
#[derive(Clone)]
pub(super) struct McpState(pub Option<Arc<llmcuda_mcp::Registry>>);
impl FromRef<AppState> for McpState {
    fn from_ref(state: &AppState) -> Self {
        Self(state.mcp.clone())
    }
}
impl McpState {
    fn registry(&self) -> Result<&Arc<llmcuda_mcp::Registry>, ApiError> {
        self.0
            .as_ref()
            .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, DIALECT, "MCP is disabled"))
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateSession {
    servers: Vec<String>,
}
pub(super) async fn create_session(
    State(state): State<McpState>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let request: CreateSession = parse_body(&body, DIALECT)?;
    let registry = state.registry()?;
    let session = registry.create(&request.servers).await.map_err(failure)?;
    Ok(Json(
        json!({"session_id":session.id, "expires_in_ms":registry.config().session_ttl_ms, "tools":session.tools()}),
    ))
}
#[derive(Deserialize)]
pub(super) struct SessionQuery {
    session_id: String,
}
pub(super) async fn list_tools(
    State(state): State<McpState>,
    Query(query): Query<SessionQuery>,
) -> Result<Json<Value>, ApiError> {
    let session = state.registry()?.get(&query.session_id).map_err(failure)?;
    let tools: Vec<_> = session.tools().into_iter().map(|tool| json!({"tool":tool.name, "server":tool.server, "original_name":tool.original_name, "definition":tool.function_definition()})).collect();
    Ok(Json(json!({"tools":tools})))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Call {
    session_id: String,
    tool: String,
    params: Map<String, Value>,
}
pub(super) async fn call_tool(
    State(state): State<McpState>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let request: Call = parse_body(&body, DIALECT)?;
    let session = state
        .registry()?
        .get(&request.session_id)
        .map_err(failure)?;
    let _lease = session.acquire().map_err(failure)?;
    let result = session
        .call(&request.tool, request.params, &session.cancellation())
        .await
        .map_err(failure)?;
    Ok(Json(
        serde_json::to_value(result).expect("MCP result serializes"),
    ))
}
pub(super) async fn close_session(
    State(state): State<McpState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    state.registry()?.remove(&id).map_err(failure)?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, to_bytes},
        http::Request,
        routing::{delete, get, post},
    };
    use tower::ServiceExt;
    fn router(state: McpState) -> Router {
        Router::new()
            .route("/tools", get(list_tools).post(call_tool))
            .route("/mcp/sessions", post(create_session))
            .route("/mcp/sessions/{id}", delete(close_session))
            .with_state(state)
    }
    async fn request(app: Router, method: &str, path: &str, body: Value) -> (StatusCode, Value) {
        let response = app
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }
    #[tokio::test]
    async fn disabled_endpoints_refuse_execution() {
        let (status, _) = request(
            router(McpState(None)),
            "POST",
            "/mcp/sessions",
            json!({"servers":["x"]}),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
    #[tokio::test]
    async fn explicit_session_list_call_close_flow() {
        let dir = std::env::temp_dir().join(format!("llmcuda-http-mcp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config=serde_json::from_value(json!({"mcpServers":{"fixture":{"command":"python3","args":["-u",format!("{}/../llmcuda-mcp/tests/fixtures/server.py",env!("CARGO_MANIFEST_DIR")),dir]}}})).unwrap();
        let app = router(McpState(Some(llmcuda_mcp::Registry::new(config).unwrap())));
        let (status, created) = request(
            app.clone(),
            "POST",
            "/mcp/sessions",
            json!({"servers":["fixture"]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{created}");
        let id = created["session_id"].as_str().unwrap();
        let (status, listed) = request(
            app.clone(),
            "GET",
            &format!("/tools?session_id={id}"),
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let name = listed["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["original_name"] == "echo")
            .unwrap()["tool"]
            .clone();
        let (status, result) = request(
            app.clone(),
            "POST",
            "/tools",
            json!({"session_id":id,"tool":name,"params":{"text":"hello"}}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(result["structuredContent"]["arguments"]["text"], "hello");
        assert_eq!(
            request(
                app.clone(),
                "DELETE",
                &format!("/mcp/sessions/{id}"),
                Value::Null
            )
            .await
            .0,
            StatusCode::NO_CONTENT
        );
        assert_eq!(
            request(app, "GET", &format!("/tools?session_id={id}"), Value::Null)
                .await
                .0,
            StatusCode::NOT_FOUND
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
