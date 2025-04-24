from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_pdf import retriever
import time
import matplotlib.pyplot as plt

# Define models with approximate parameter counts
models = [
    {"name": "llama3.2", "params": "Unknown", "instance": OllamaLLM(model="llama3.2")},
    {"name": "llama3.2:1b", "params": "1B", "instance": OllamaLLM(model="llama3.2:1b")},
    {"name": "deepseek-r1:1.5b", "params": "1.5B", "instance": OllamaLLM(model="deepseek-r1:1.5b")}
]

# Define a list of five common questions for the experiment
common_questions = [
    "What is the overall summary of the document?",
    "What are the key findings presented?",
    "How does the document relate to current trends in the field?",
    "What implications can be drawn from the document's conclusions?",
    "Can you provide a critical analysis of the document?"
]

# Prompt template remains the same
template = """
You are an expert in answering questions, Answer in just 50 words.

Here are some relevant pdfs: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Dictionary to store execution times and answers for each model
execution_times = {
    model["name"]: {"retrieval": [], "answer": []} for model in models
}
model_answers = { model["name"]: {} for model in models }

# Loop through each model and then through each question
for model_info in models:
    model_name = model_info["name"]
    print(f"\n===== Testing model: {model_name} (Params: {model_info['params']}) =====")
    
    # Build a chain for the current model using the prompt and model instance
    chain = prompt | model_info["instance"]
    
    # For each question, measure the retrieval and answer generation time
    for idx, question in enumerate(common_questions, start=1):
        print(f"\nQuestion {idx}: {question}")
        
        # Time the retrieval process
        start_retrieval = time.time()
        reviews = retriever.invoke(question)
        end_retrieval = time.time()
        retrieval_time = end_retrieval - start_retrieval
        execution_times[model_name]["retrieval"].append(retrieval_time)
        print(f"Retrieval time: {retrieval_time:.4f} seconds")
        
        # Time the answer generation process
        start_answer = time.time()
        result = chain.invoke({"reviews": reviews, "question": question})
        end_answer = time.time()
        answer_time = end_answer - start_answer
        execution_times[model_name]["answer"].append(answer_time)
        print(f"Answer generation time: {answer_time:.4f} seconds")
        
        # Store the answer for later analysis
        model_answers[model_name][question] = result

# Plot the time profiles for each model
def plot_times():
    plt.figure(figsize=(12, 8))
    
    for model_name, times in execution_times.items():
        iterations = list(range(1, len(times["retrieval"]) + 1))
        plt.plot(iterations, times["retrieval"], marker='o', label=f"{model_name} - Retrieval")
        plt.plot(iterations, times["answer"], marker='x', label=f"{model_name} - Answer")
    
    plt.title("Execution Times per Question for Each Model")
    plt.xlabel("Question Iteration")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.savefig("model_time_profile.png")
    plt.show()

plot_times()

# Optionally, print the answers for review
for model_name, answers in model_answers.items():
    print(f"\n==== Answers from {model_name} ====")
    for q, ans in answers.items():
        print(f"\nQ: {q}\nA: {ans}\n")
