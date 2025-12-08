from src.llm_engine import LLMEngine
import time

def run_test():
    llm = LLMEngine()
    
    print("\n\n=== TEST 1: Saving a Fact ===")
    user_input_1 = "Friday, I started working out from today."
    print(f"User: {user_input_1}")
    print("Friday: ", end="")
    for chunk in llm.generate_response_stream(user_input_1):
        print(chunk, end="", flush=True)
    
    print("\n\n=== TEST 2: Complex Retrieval & Reasoning ===")
    user_input_2 = "Friday, Recommend me a diet plan and consider the time since I started working out."
    print(f"User: {user_input_2}")
    print("Friday: ", end="")
    for chunk in llm.generate_response_stream(user_input_2):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    run_test()
