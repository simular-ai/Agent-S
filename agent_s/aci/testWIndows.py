# experiment_windowsACI.py

from windowsOSACI import WindowsACI

def execute_command(command_str):
    # Execute the command string returned by the ACI methods
    try:
        exec(command_str, globals(), locals())
    except Exception as e:
        print(f"Error executing command: {e}")

def main():
    # Create an instance of WindowsACI
    aci = WindowsACI(top_app_only=True, ocr=False)

    # Open Notepad
    print("Opening Notepad...")
    command = aci.open('notepad.exe')
    execute_command(command)

    # Wait for Notepad to open
    command = aci.wait(2)
    execute_command(command)

    # Get the current accessibility tree
    obs = {}
    tree = aci.linearize_and_annotate_tree(obs)
    print("\nAccessibility Tree:")
    print(tree)

    # Assuming the text area has element_id 0 (this may vary)
    element_id = 0

    # Type some text into Notepad
    print("\nTyping text into Notepad...")
    command = aci.type(element_id=element_id, text="Hello, world!", overwrite=False, enter=False)
    execute_command(command)

    # Wait a bit
    command = aci.wait(1)
    execute_command(command)

    # Save the file using Ctrl+S
    print("\nSaving the file...")
    command = aci.hotkey(['ctrl', 's'])
    execute_command(command)

    # Wait for the Save dialog to appear
    command = aci.wait(2)
    execute_command(command)

    # Type the filename and save
    command = aci.type(text="test_document.txt", enter=True)
    execute_command(command)

    # Wait a bit
    command = aci.wait(1)
    execute_command(command)

    # Close Notepad using Alt+F4
    print("\nClosing Notepad...")
    command = aci.hotkey(['alt', 'f4'])
    execute_command(command)

    # Wait a bit
    command = aci.wait(1)
    execute_command(command)

    # Indicate that the task is done
    command = aci.done()
    print(f"\nTask Status: {command}")

if __name__ == "__main__":
    main()
