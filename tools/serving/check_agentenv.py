"""Live AgentENV checks; creates two sandboxes through the configured MCP bridge.

The selected template must contain Python and Node. Sessions are closed after
checks; the bridge's remote lifetime bounds cleanup after forced process exit.
"""
import argparse
import json
import os
from pathlib import Path
import uuid

from check_mcp import Client


def command_output(result):
    assert not result.get('isError'), result
    if 'structuredContent' in result:
        return result['structuredContent']
    return json.loads(''.join(block.get('text', '') for block in result['content']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--api-key-env', default='LLMXABE_API_KEY')
    parser.add_argument('--server', default='agentenv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    client = Client(args.candidate, os.environ[args.api_key_env])
    sessions=[]
    report={}
    def save(name,value):
        report[name]=value
        args.output.write_text(json.dumps(report, indent=2))
        print('PASS:',name,flush=True)
    try:
        first=client.request('/mcp/sessions',{'servers':[args.server]}); sessions.append(first['session_id'])
        second=client.request('/mcp/sessions',{'servers':[args.server]}); sessions.append(second['session_id'])
        alias=first['tools'][0]['name']
        path='/tmp/llmxabe-mcp-'+uuid.uuid4().hex
        result=client.request('/tools',{'session_id':sessions[0],'tool':alias,'params':{'command':f"printf 'session-one' > {path}; cat {path}"}})
        assert not result.get('isError') and command_output(result)['stdout']=='session-one',result
        isolated=client.request('/tools',{'session_id':sessions[1],'tool':second['tools'][0]['name'],'params':{'command':f'test ! -e {path} && echo isolated'}})
        assert not isolated.get('isError') and command_output(isolated)['stdout'].strip()=='isolated',isolated
        save('explicit_command_and_session_isolation',{'first':result,'second':isolated})
        runtimes=client.request('/tools',{'session_id':sessions[0],'tool':alias,'params':{'command': 'python3 -c \"print(6 * 7)\" && node -e \"console.log(6 * 7)\"'}})
        assert not runtimes.get('isError') and command_output(runtimes)['stdout'].split()==['42','42'],runtimes
        save('python_and_node',runtimes)
        command=f"cat /proc/sys/kernel/random/uuid > {path}; cat {path}"
        task=f'Use {alias} exactly once to run this command: {command}\nReturn only the UUID printed by the command. Do not guess it.'
        body={'input':task,'temperature':0,'max_output_tokens':512,'reasoning':{'effort':'none'},'mcp':{'session_id':sessions[0],'max_iterations':3}}
        answer=client.request('/v1/responses',body)
        assert answer['status']=='completed',answer
        value=command_output(client.request('/tools',{'session_id':sessions[0],'tool':alias,'params':{'command':f'cat {path}'}}))['stdout'].strip()
        uuid.UUID(value)
        assert value in json.dumps(answer['output']),answer
        save('gpu_agentenv_generate_execute_resume',answer)
        body['input']=[{'role':'user','content':task}]+answer['mcp_history']+[{'role':'user','content':f'Use the tool once to read {path} again, then return only its contents.'}]
        events=client.stream('/v1/responses',body)
        assert events[-1]['type']=='response.completed' and value in json.dumps(events[-1]['response']['output']),events
        assert any(e.get('event',{}).get('type')=='tool_completed' for e in events),events
        save('gpu_agentenv_streamed_followup',events)
    finally:
        for sid in sessions:
            client.request('/mcp/sessions/'+sid,method='DELETE')
    print('PASS: real AgentENV sandbox execution, isolation, model loop and streaming replay',flush=True)


if __name__ == "__main__":
    main()
