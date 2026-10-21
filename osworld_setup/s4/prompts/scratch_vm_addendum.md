## Scratch VM

You have a fresh AWS OSWorld VM available for this mutation only. It boots
with the parent interface pre-deployed at /home/user/shadow_interface/ and
is destroyed when you exit. Use it to explore apps, study source code, try
the parent's interface live, and (once you have a hypothesis) test your own
code by redeploying.

Access it via the framework-owned scratch bridge.
Run bridge code with the interpreter in `$SCRATCH_VM_PYTHON` (for example
`"$SCRATCH_VM_PYTHON" -c '...'` or a script under `.builder_tmp/`). It is the
one that has the bridge's dependencies and that the sandbox lets you execute;
your own Python tool and the `python` on PATH may lack them.

    from scratch_vm import get_controller, get_setup_controller, redeploy_interface

    c = get_controller()
    c.get_screenshot()                     # bytes
    c.get_accessibility_tree()             # XML string
    c.execute_python_command("...")        # run any Python on the VM

    sc = get_setup_controller()
    sc.setup([{"type": "launch", "parameters": {"command": ["libreoffice","--calc"]}}])

    # When ready to test new code you just wrote in shadow_interface/
    redeploy_interface()
    c.execute_python_command("...")        # now hits your new code

Exploration output (screenshots, a11y dumps) dies with the VM — capture any
lasting insights in notebook.jsonl before you finish.

Strategy: explore first, form a hypothesis from what apps offer, THEN write
the mutation. Iterating code-test-code on the VM is expensive; use it for
validation not blind search. Formal OSWorld V2 runs fail closed before the
Builder starts if the scratch VM is unavailable. If importing `scratch_vm`
still raises ScratchVmUnavailable, stop and report the infrastructure error;
do not silently continue as a trajectory-only run.
