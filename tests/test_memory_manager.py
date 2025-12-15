from src.llm_engine import LLMEngine
import time

def run_test():
    llm = LLMEngine()
    
    print("\n\n=== TEST 1: Saving a Fact ===")
    user_input_1 = "Emma, I am a very fickle person."
    print(f"User: {user_input_1}")
    print("Friday: ", end="")
    full_response = ""
    for chunk in llm.generate_response_stream(user_input_1):
        print(chunk, end="", flush=True)
        full_response += chunk
    print("\n(Memory should be saved after this)")
    
    # Wait a sec for DB to persist if async (it's sync here but good practice)
    time.sleep(1) 
    
    print("\n\n=== TEST 2: Retrieval ===")
    user_input_2 = "Emma, What kind of person am I?"
    print(f"User: {user_input_2}")
    print("Friday: ", end="")
    for chunk in llm.generate_response_stream(user_input_2):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    run_test()
