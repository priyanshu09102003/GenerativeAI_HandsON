from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

#Example: Python code that is to be split/chunked
text_code = """

    class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


    # Example usage
    student1 = Student("Aarav", 20, 8.2)
    print(student1.get_details())

    if student1.is_passing():
        print("The student is passing.")
    else:
        print("The student is not passing.")

"""

text_markdown = """
    # Project Name: Smart Student Tracker

    A simple Python-based project to manage and track student data, including their grades, age, and academic status.


    ## Features

    - Add new students with relevant info
    - View student details
    - Check if a student is passing
    - Easily extendable class-based design


    ## 🛠 Tech Stack

    - Python 3.10+
    - No external dependencies


    ## Getting Started

    1. Clone the repo  
    ```bash
    git clone https://github.com/your-username/student-tracker.git
"""


splitter_python = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0,
)

splitter_markdown = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 200,
    chunk_overlap = 0
)

chunk_python = splitter_python.split_text(text_code)
chunk_markdown = splitter_markdown.split_text(text_markdown)

print()
print("Chunked Python Code:")
print()
print(chunk_python)
print()
print("Chunked Markdown:")
print()
print(chunk_markdown)