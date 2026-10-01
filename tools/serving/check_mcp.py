"""Live MCP and ordinary tool-call checks against a running llmcuda-rs.

Start the server with an API key and MCP labels `echo`, `error`, and `slow`,
pointing at crates/llmcuda-mcp/tests/fixtures/server.py and allowing the respective
tool. This script executes only those fixture tools. Reports belong outside git.
"""
import argparse
import json
import os
import pathlib
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, base, key):
        self.base, self.key = base.rstrip('/'), key

    def open(self, path, body=None, method=None, auth=True):
        headers = {'Content-Type': 'application/json'}
        if auth:
            headers['Authorization'] = 'Bearer ' + self.key
        request = urllib.request.Request(self.base + path,
            None if body is None else json.dumps(body).encode(), headers, method=method)
        return urllib.request.urlopen(request, timeout=180)

    def request(self, path, body=None, method=None):
        with self.open(path, body, method) as response:
            data = response.read()
            return json.loads(data) if data else None

    def stream(self, path, body):
        with self.open(path, dict(body, stream=True)) as response:
            return [json.loads(line[6:]) for line in response
                    if line.startswith(b'data: ') and line[6:].strip() != b'[DONE]']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--api-key-env', default='LLMCUDA_API_KEY')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--fixture-events', type=pathlib.Path, required=True,
                        help='Event directory passed to the stdio fixture')
    args = parser.parse_args()
    client = Client(args.candidate, os.environ[args.api_key_env])
    report = {}

    def record(name, value):
        report[name] = value
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print('PASS:', name, flush=True)

    try:
        client.open('/mcp/sessions', {'servers': ['echo']}, auth=False)
        raise AssertionError('MCP session creation must require authentication')
    except urllib.error.HTTPError as error:
        assert error.code == 401
    record('authentication', 401)

    schema = {'type': 'object', 'properties': {'text': {'type': 'string'},
        'count': {'type': 'integer'}, 'enabled': {'type': 'boolean'}},
        'required': ['text', 'count', 'enabled']}
    function = {'name': 'record_values', 'description': 'Record the exact supplied values.',
                'parameters': schema}
    prompt = 'Call record_values once with text="你好", count=7, enabled=false. Do not answer until you get the tool result. Then report its verification_code exactly.'
    expected = {'text': '你好', 'count': 7, 'enabled': False}
    chat = {'model': 'test', 'temperature': 0, 'max_tokens': 256,
        'messages': [{'role': 'user', 'content': prompt}],
        'chat_template_kwargs': {'enable_thinking': False},
        'tools': [{'type': 'function', 'function': function}]}
    response = client.request('/v1/chat/completions', chat)
    call = response['choices'][0]['message']['tool_calls'][0]
    assert response['choices'][0]['finish_reason'] == 'tool_calls'
    assert json.loads(call['function']['arguments']) == expected
    code = 'verified-chat-7192'
    replay = dict(chat, messages=chat['messages'] + [response['choices'][0]['message'],
        {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps({'verification_code': code})}])
    answer = client.request('/v1/chat/completions', replay)
    assert code in answer['choices'][0]['message']['content']
    streamed = client.stream('/v1/chat/completions', chat)
    calls = [c for e in streamed for choice in e.get('choices', []) for c in choice.get('delta', {}).get('tool_calls', [])]
    assert json.loads(''.join(c['function'].get('arguments', '') for c in calls)) == expected
    record('chat_tools_and_replay', {'response': response, 'answer': answer, 'stream': streamed})

    anthropic = {'model': 'test', 'temperature': 0, 'max_tokens': 256,
        'thinking': {'type': 'disabled'}, 'messages': chat['messages'],
        'tools': [{'name': function['name'], 'description': function['description'], 'input_schema': schema}]}
    response = client.request('/v1/messages', anthropic)
    call = next(b for b in response['content'] if b['type'] == 'tool_use')
    assert response['stop_reason'] == 'tool_use' and call['input'] == expected
    code = 'verified-anthropic-8213'
    replay = dict(anthropic, messages=anthropic['messages'] + [{'role': 'assistant', 'content': response['content']},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call['id'], 'content': json.dumps({'verification_code': code})}]}])
    answer = client.request('/v1/messages', replay)
    assert code in ''.join(b.get('text', '') for b in answer['content'])
    streamed = client.stream('/v1/messages', anthropic)
    arguments = ''.join(e.get('delta', {}).get('partial_json', '') for e in streamed)
    assert json.loads(arguments) == expected
    record('anthropic_tools_and_replay', {'response': response, 'answer': answer, 'stream': streamed})

    responses = {'model': 'test', 'temperature': 0, 'max_output_tokens': 256,
        'reasoning': {'effort': 'none'}, 'input': prompt,
        'tools': [dict(type='function', **function)]}
    response = client.request('/v1/responses', responses)
    call = next(i for i in response['output'] if i['type'] == 'function_call')
    assert json.loads(call['arguments']) == expected
    code = 'verified-responses-9124'
    replay = dict(responses, input=[{'role': 'user', 'content': prompt}] + response['output'] +
        [{'type': 'function_call_output', 'call_id': call['call_id'], 'output': json.dumps({'verification_code': code})}])
    answer = client.request('/v1/responses', replay)
    assert code in json.dumps(answer['output'])
    streamed = client.stream('/v1/responses', responses)
    terminal = next(e['response'] for e in streamed if e['type'] == 'response.completed')
    assert json.loads(next(i for i in terminal['output'] if i['type'] == 'function_call')['arguments']) == expected
    record('responses_tools_and_replay', {'response': response, 'answer': answer, 'stream': streamed})

    session = client.request('/mcp/sessions', {'servers': ['echo']})
    sid = session['session_id']
    try:
        catalog = client.request('/tools?session_id=' + sid)
        alias = catalog['tools'][0]['tool']
        explicit = client.request('/tools', {'session_id': sid, 'tool': alias, 'params': {'text': 'nested 你好', 'nested': {'a': [True, 2]}}})
        assert explicit['structuredContent']['arguments']['nested'] == {'a': [True, 2]}
        pid = str(explicit['structuredContent']['pid'])
        task = f'Call {alias} exactly once with text="live-check". From its tool result read the pid field and answer with only that number. Do not guess the pid.'
        body = {'input': task, 'temperature': 0, 'max_output_tokens': 512,
            'reasoning': {'effort': 'none'}, 'mcp': {'session_id': sid, 'max_iterations': 3}}
        response = client.request('/v1/responses', body)
        assert response['status'] == 'completed' and response['mcp_stop_reason'] == 'end_turn', response
        assert pid in json.dumps(response['output'])
        history = response['mcp_history']
        assert len([i for i in history if i.get('type') == 'function_call']) == 1
        assert len([i for i in history if i.get('type') == 'function_call_output']) == 1
        follow = dict(body, input=[{'role': 'user', 'content': task}] + history +
            [{'role': 'user', 'content': 'Without calling any tool, repeat the pid from the previous tool result. Only the number.'}])
        replay = client.request('/v1/responses', follow)
        assert pid in json.dumps(replay['output'])
        streamed = client.stream('/v1/responses', body)
        assert [e['sequence_number'] for e in streamed] == list(range(len(streamed)))
        events = [e['event']['type'] for e in streamed if e['type'] == 'response.mcp_event']
        assert events == ['model_turn', 'tool_started', 'tool_completed', 'model_turn'], events
        assert pid in json.dumps(streamed[-1]['response']['output'])
        limited = client.request('/v1/responses', dict(body, mcp={'session_id': sid, 'max_iterations': 1}))
        assert limited['status'] == 'incomplete' and limited['mcp_stop_reason'] == 'max_iterations'
        assert not any(i.get('type') == 'function_call_output' for i in limited['mcp_history'])
        record('mcp_loop_stream_replay_and_limits', {'explicit': explicit, 'response': response, 'replay': replay, 'stream': streamed, 'limited': limited})
    finally:
        client.request('/mcp/sessions/' + sid, method='DELETE')

    session = client.request('/mcp/sessions', {'servers': ['error']})
    sid, alias = session['session_id'], session['tools'][0]['name']
    try:
        response = client.request('/v1/responses', {
            'input': f'Call {alias} exactly once with text="error-check". Then report whether it succeeded or failed. Never retry.',
            'temperature': 0, 'max_output_tokens': 512, 'reasoning': {'effort': 'none'},
            'mcp': {'session_id': sid, 'max_iterations': 3}})
        results = [i for i in response['mcp_history'] if i.get('type') == 'function_call_output']
        assert response['status'] == 'completed' and len(results) == 1, response
        assert 'operation failed' in json.dumps(results) and 'fail' in json.dumps(response['output']).lower(), response
        record('mcp_tool_error_resume', response)
    finally:
        client.request('/mcp/sessions/' + sid, method='DELETE')

    # Cancellation exercises the real model followed by a slow external tool.
    session = client.request('/mcp/sessions', {'servers': ['slow']})
    sid, alias = session['session_id'], session['tools'][0]['name']
    try:
        marker = f'cancel-check-{time.monotonic_ns()}'
        body = {'input': f'Call {alias} exactly once with text="{marker}". Wait for its result.', 'stream': True,
            'temperature': 0, 'max_output_tokens': 256, 'reasoning': {'effort': 'none'},
            'mcp': {'session_id': sid, 'max_iterations': 3}}
        events = []
        with client.open('/v1/responses', body) as stream:
            for line in stream:
                if not line.startswith(b'data: '):
                    continue
                event = json.loads(line[6:]); events.append(event)
                if event.get('event', {}).get('type') == 'tool_started':
                    deadline = time.monotonic() + 5
                    while not any(marker in p.read_text() and '"tools/call"' in p.read_text()
                                  for p in args.fixture_events.glob('events-*.jsonl')):
                        assert time.monotonic() < deadline, 'tool never reached the fixture'
                        time.sleep(0.02)
                    client.request('/mcp/sessions/' + sid, method='DELETE')
                if event['type'] == 'error':
                    break
        assert events[-1]['type'] == 'error' and 'cancel' in events[-1]['error']['message']
        record('mcp_cancel_active_tool', events)
    finally:
        try:
            client.request('/mcp/sessions/' + sid, method='DELETE')
        except urllib.error.HTTPError as error:
            assert error.code == 404
    print('PASS: live GPU tool calls, MCP execution/replay/streaming/limits/cancellation', flush=True)


if __name__ == '__main__':
    main()
