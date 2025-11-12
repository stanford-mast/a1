# Complete Context Append Analysis

## All Locations Where Messages Are Appended to Context

### 1. runtime.py - JIT Execution

**Line 488-489**: JIT starts
```python
context.user(user_message)           # Main context - user input
temp_context.user(user_message)      # Temp context - user input
```
- **What**: User input at start of JIT
- **Main Context**: ✅ YES
- **Temp Context**: ✅ YES
- **Attached to Runtime**: ✅ YES (get_context("main"))
- **Timestamp**: ❌ NO

**Line 601-603**: Execution retry failures
```python
temp_context.assistant(f"```python\n{code}\n```")    # Failed code
temp_context.system(f"Execution error: {exec_result.error}")  # Error
```
- **What**: Failed execution attempts
- **Main Context**: ❌ NO (only temp)
- **Temp Context**: ✅ YES
- **Attached to Runtime**: ❌ NO (temp_context is local variable, not in CTX)
- **Timestamp**: ❌ NO
- **Issue**: Temp context is NOT attached to runtime, gets discarded

**Line 622**: JIT success
```python
context.assistant(output_str)        # Successful output
```
- **What**: Successful JIT output
- **Main Context**: ✅ YES
- **Temp Context**: ❌ NO
- **Attached to Runtime**: ✅ YES
- **Timestamp**: ❌ NO

### 2. runtime.py - execute() for Non-LLM Tools

**Line 700-703**: Tool call start
```python
context.assistant(
    content="",
    tool_calls=[...]
)
```
- **What**: Assistant calling a non-LLM tool
- **Main Context**: ✅ YES
- **Temp Context**: ❌ NO
- **Attached to Runtime**: ✅ YES (get_context("main"))
- **Timestamp**: ❌ NO

**Line 719-724**: Tool success
```python
context.tool(
    content=result_str,
    name=tool.name,
    tool_call_id=tool_call_id
)
```
- **What**: Non-LLM tool result (success)
- **Main Context**: ✅ YES
- **Temp Context**: ❌ NO
- **Attached to Runtime**: ✅ YES
- **Timestamp**: ❌ NO

**Line 733-737**: Tool error
```python
context.tool(
    content=f"Error: {str(e)}",
    name=tool.name,
    tool_call_id=tool_call_id
)
```
- **What**: Non-LLM tool result (error)
- **Main Context**: ✅ YES
- **Temp Context**: ❌ NO
- **Attached to Runtime**: ✅ YES
- **Timestamp**: ❌ NO

### 3. llm.py - LLM Tool Execution

**Line 288**: LLM user input
```python
context.user(content)
```
- **What**: User content to LLM
- **Main Context**: Depends (context parameter)
- **Temp Context**: ❌ NO
- **Attached to Runtime**: ⚠️ MAYBE (depends if context is from get_context or no_context)
- **Timestamp**: ❌ NO

**Line 366-368**: LLM response
```python
context.assistant(response_content, tool_calls=tool_call_dicts)  # With tools
context.assistant(response_content)                               # Without tools
```
- **What**: LLM response
- **Main Context**: Depends (context parameter)
- **Temp Context**: ❌ NO
- **Attached to Runtime**: ⚠️ MAYBE
- **Timestamp**: ❌ NO

**Line 401-405, 471-475**: LLM tool results
```python
context.tool(
    content=result_str,
    name=func_name,
    tool_call_id=tool_call.id
)
```
- **What**: Results from tools called by LLM
- **Main Context**: Depends (context parameter)
- **Temp Context**: ❌ NO
- **Attached to Runtime**: ⚠️ MAYBE
- **Timestamp**: ❌ NO

## All Locations Where Contexts Are Created

### 1. runtime.py

**Line 475**: JIT main context
```python
self.CTX["main"] = Context()
```
- **Attached to Runtime**: ✅ YES (in CTX dict)
- **Named**: ✅ YES ("main")
- **Timestamp**: ❌ NO

**Line 484**: JIT temp context  
```python
temp_context = Context()
```
- **Attached to Runtime**: ❌ NO (local variable)
- **Named**: ❌ NO
- **Timestamp**: ❌ NO
- **Issue**: This is discarded after JIT, not accessible for debugging

**Line 678**: execute() main context
```python
self.CTX["main"] = Context()
```
- **Attached to Runtime**: ✅ YES
- **Named**: ✅ YES ("main")
- **Timestamp**: ❌ NO

**Line 957**: get_context() creates new
```python
ctx = Context()
```
- **Attached to Runtime**: ✅ YES (immediately added to CTX)
- **Named**: ✅ YES (via key parameter)
- **Timestamp**: ❌ NO

### 2. llm.py

**Line 32**: no_context()
```python
return Context()
```
- **Attached to Runtime**: ❌ NO
- **Named**: ❌ NO
- **Timestamp**: ❌ NO
- **Purpose**: Throwaway context for one-off LLM calls
- **Issue**: Not tracked anywhere, can't access history

**Line 235**: LLM execute with no context
```python
context = no_context()
```
- **Attached to Runtime**: ❌ NO
- **Named**: ❌ NO
- **Timestamp**: ❌ NO
- **Issue**: Same as above

### 3. executor.py

**Line 242**: Executor environment
```python
exec_env['CTX'] = {'main': Context()}
```
- **Attached to Runtime**: ❌ NO (only in exec environment)
- **Named**: ✅ YES ("main")
- **Timestamp**: ❌ NO
- **Purpose**: For code execution environment
- **Issue**: Separate from runtime CTX

### 4. context.py

**Line 138**: no_history() function
```python
return Context()
```
- **Attached to Runtime**: ❌ NO
- **Named**: ❌ NO
- **Timestamp**: ❌ NO
- **Purpose**: Throwaway context
- **Issue**: Same as no_context()

## Summary of Issues

### Critical Issues:

1. **No Timestamps**: Messages don't have timestamps
   - Can't tell when messages were created
   - Can't interleave contexts chronologically
   - Can't debug timing issues

2. **Temp Contexts Not Attached**: 
   - `temp_context` in JIT is a local variable
   - Gets discarded after execution
   - Can't access for debugging/observability
   - Failed attempts are lost

3. **no_context() Not Tracked**:
   - Creates throwaway contexts
   - Not attached to runtime
   - Can't access history later
   - Multiple calls create multiple isolated contexts

4. **Multiple Context Sources**:
   - Runtime.CTX (main contexts)
   - Executor exec_env['CTX'] (for code execution)
   - Local temp_context variables
   - no_context() throwaways
   - No unified view

5. **No get_full_context()**:
   - Can't get interleaved timeline of all contexts
   - Can't see full conversation across contexts
   - Can't debug multi-context interactions

### Proposed Solutions:

1. **Add Timestamps to Message**:
   ```python
   class Message(BaseModel):
       role: str
       content: str
       timestamp: datetime = Field(default_factory=datetime.now)
       ...
   ```

2. **Register Temp Contexts**:
   ```python
   # In Runtime.jit()
   temp_key = f"_temp_{uuid.uuid4().hex[:8]}"
   self.CTX[temp_key] = temp_context
   ```

3. **Fix no_context()**:
   ```python
   def no_context():
       runtime = get_runtime()
       temp_key = f"_tmp_{len([k for k in runtime.CTX if k.startswith('_tmp_')])}"
       ctx = Context()
       runtime.CTX[temp_key] = ctx
       return ctx
   ```

4. **Add get_full_context()**:
   ```python
   def get_full_context() -> List[Message]:
       runtime = get_runtime()
       all_messages = []
       for ctx in runtime.CTX.values():
           all_messages.extend(ctx.messages)
       all_messages.sort(key=lambda m: m.timestamp)
       return all_messages
   ```

## Recommended Changes:

1. ✅ Add `timestamp` to Message model
2. ✅ Change `no_context()` to register in Runtime with auto-incrementing `_tmp_0`, `_tmp_1`, etc.
3. ✅ Register `temp_context` in Runtime during JIT (as `_temp_jit_{id}`)
4. ✅ Add `Runtime.get_full_context()` to get all messages sorted by timestamp
5. ✅ Add `Runtime.get_context_names()` to list all context names
6. ✅ Add `Runtime.get_temp_contexts()` to get only temp contexts for debugging
