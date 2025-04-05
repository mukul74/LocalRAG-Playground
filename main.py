from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# from vector_csv import retriever
from vector_pdf import retriever
import time
import matplotlib.pyplot as plt

# To store execution times
execution_times = {
    "retrive": [],
    "answer": []
}

def time_it(func_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()  # Record the start time
            result = func(*args, **kwargs)  # Call the actual function
            end_time = time.time()  # Record the end time
            execution_times[func_name].append(end_time - start_time)  # Store the execution time
            print(f"Function '{func_name}' took {end_time - start_time:.4f} seconds to execute.")
            return result
        return wrapper
    return decorator


model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions.

Here are some relevant pdfs: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@time_it("retrive")
def retrive(question):
    # This function retrieves relevant documents based on the question
    # and returns them as a string.
    reviews = retriever.invoke(question)
    return reviews

@time_it("answer")
def answer(reviews, question):
    # This function takes a question and retrieves relevant documents,
    # then generates an answer using the language model.
    result = chain.invoke({"reviews": reviews, "question": question})
    return result

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    retrived = retrive(question)
    result = answer(retrived, question)

    print(result)

# After the loop ends, plot the time profile for each function
def plot_times():
    # Plot the time profile for both functions
    plt.figure(figsize=(10, 6))

    # Plotting retrieval times
    plt.plot(execution_times["retrive"], label="Retrive Function Time", marker='o')
    # Plotting answering times
    plt.plot(execution_times["answer"], label="Answer Function Time", marker='o')

    plt.title("Execution Times of Functions")
    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("my_plot.png")

# Call the plot function to display the graph after exiting the loop
plot_times()
