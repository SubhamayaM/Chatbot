from llama_cpp import Llama
import time

print("Loading model...")
llm = Llama(
    model_path="./models/llama/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=512,
    n_threads=4,
    verbose=True
)
print("Model loaded!")

prompt = "Q: What is the capital of France?\nA:"
print("Running prompt...")
start = time.time()
output = llm(prompt, max_tokens=20)
end = time.time()

print("Completed in", round(end - start, 2), "seconds")
print("Model says:", output["choices"][0]["text"].strip())
