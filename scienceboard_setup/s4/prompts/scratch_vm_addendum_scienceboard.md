## Scratch VM

You have a fresh AWS ScienceBoard VM available for this build only. It boots
with the parent interface pre-deployed at /home/user/shadow_interface/ and is
destroyed when you exit. The scientific applications (Celestia, ChimeraX,
GrassGIS, KAlgebra, Lean, TeXstudio) are installed. Use it to explore the apps,
study source code, try the parent's interface live, and (once you have a
hypothesis) test your own code by redeploying.

Access it via the framework-owned scratch bridge.
Run bridge code with the interpreter in `$SCRATCH_VM_PYTHON` (for example
`"$SCRATCH_VM_PYTHON" -c '...'` or a script under `.builder_tmp/`). It is the
one that has the bridge's dependencies and that the sandbox lets you execute;
your own Python tool and the `python` on PATH may lack them.

    from scratch_vm import get_controller, redeploy_interface

    c = get_controller()
    c.get_screenshot()                     # bytes
    c.get_accessibility_tree()             # XML string
    c.execute_python_command("...")        # run any Python on the VM

There is no pre-launched app and no SetupController: you launch and drive the
apps yourself through execute_python_command. Each app exposes a control server
(for example, ChimeraX is started with a flag like
`--cmd "remotecontrol rest start json true port <P>"`). Launch them as needed,
or inspect what is already running with `ps aux` on the VM to discover the exact
command and ports.

    # When ready to test new code you just wrote in shadow_interface/
    redeploy_interface()
    c.execute_python_command("...")        # now hits your new code

Exploration output (screenshots, a11y dumps) dies with the VM — capture any
lasting insights in notebook.jsonl before you finish.

Strategy: explore first, form a hypothesis from what the apps offer, THEN write
the change. Iterating code-test-code on the VM is expensive; use it for
validation not blind search. If the scratch VM is unavailable (rare
provisioning failure), importing `scratch_vm` raises
ScratchVmUnavailable — fall back to working from eval_results/.
