import tkinter as tk
import subprocess

def execute_script_1():
    name = text_entry.get()  # Get the text entered by the user
    subprocess.Popen(['python', 'headshot.py', name])

def execute_script_2():
    subprocess.Popen(['python', 'recognition.py'])

def exit_window():
    root.destroy()

# Create the Tkinter window
root = tk.Tk()
root.title("Face Recog")
root.geometry("400x300")

# Create the heading label
heading_label = tk.Label(root, text="Face Recognition", font=("Helvetica", 24))
heading_label.pack(pady=20)

# Create the name entry label and widget
name_label = tk.Label(root, text="Enter your name without any spaces:")
name_label.pack()

text_entry = tk.Entry(root, font=("Helvetica", 12))
text_entry.pack(pady=10)

# Create the buttons
button_frame = tk.Frame(root)
button_frame.pack()

button1 = tk.Button(button_frame, text="Headshot ('s': click, 'q': quit)", command=execute_script_1)
button1.pack(side=tk.LEFT, padx=10)

button2 = tk.Button(button_frame, text="Recognition ('q': quit)", command=execute_script_2)
button2.pack(side=tk.LEFT, padx=10)

exit_button = tk.Button(root, text="Exit", command=exit_window)
exit_button.pack(pady=20)

# Start the Tkinter event loop
root.mainloop()
