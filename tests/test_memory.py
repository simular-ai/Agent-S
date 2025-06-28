from gui_agents.s2.memory.memory import Memory

def run_tests():
    mem = Memory("test_memory.json")

    print("Storing task...")
    mem.store("test_task", {
        "description": "open notepad",
        "steps": ["click start", "type notepad", "press enter"]
    })

    print("Retrieving task...")
    data = mem.retrieve("test_task")
    print("Retrieved:", data)

    print("Updating task...")
    mem.update("test_task", {
        "status": "completed",
        "result": "notepad opened"
    })

    updated = mem.retrieve("test_task")
    print("Updated:", updated)

    print("Clearing task...")
    mem.clear("test_task")

    cleared = mem.retrieve("test_task")
    print("After clear:", cleared)

if __name__ == "__main__":
    run_tests()
