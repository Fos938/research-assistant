import os
import json
from datetime import datetime
from openai import OpenAI
import time
import argparse

class ResearchAssistant:
    def __init__(self, api_key=None):
        """Initialize the research assistant with API key and configuration."""
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as PERPLEXITY_API_KEY environment variable")
            
        # Initialize the OpenAI client with Perplexity base URL
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url="https://api.perplexity.ai"
        )
        
        # Default system prompt for research
        self.default_system_prompt = (
            "You are an elite research assistant with exceptional analytical abilities. "
            "Your task is to provide comprehensive, well-structured, and accurate information "
            "on any topic. For each query:"
            "\n1. Break down complex topics into digestible components"
            "\n2. Cite specific sources where possible"
            "\n3. Distinguish between facts and speculation"
            "\n4. Identify knowledge gaps and limitations"
            "\n5. Organize information in a logical progression"
            "\nWhen appropriate, include relevant statistics, expert viewpoints, "
            "historical context, and current developments."
        )
        
        # Research session tracking
        self.session_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = "research_sessions"
        
    def research(self, query, system_prompt=None, model="sonar-reasoning-pro", stream=False, temperature=0.7):
        """Conduct research on a specific query"""
        
        # Use default system prompt if none provided
        if system_prompt is None:
            system_prompt = self.default_system_prompt
            
        # Prepare messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Add all previous exchanges to create context
        if self.session_history:
            messages = messages[:1] + self.session_history + messages[1:]
        
        try:
            start_time = time.time()
            
            if stream:
                # Streaming response
                response_stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
                
                # Print streaming response as it comes in
                collected_chunks = []
                print("\n--- Research Results ---\n")
                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        collected_chunks.append(content)
                
                # Join chunks to get complete response
                complete_response = "".join(collected_chunks)
                print("\n\n")
                
            else:
                # Non-streaming response
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                complete_response = response.choices[0].message.content
                print("\n--- Research Results ---\n")
                print(complete_response)
                print("\n")
            
            # Calculate time taken
            time_taken = time.time() - start_time
            print(f"Research completed in {time_taken:.2f} seconds")
            
            # Update session history with this exchange
            self.session_history.append({"role": "user", "content": query})
            self.session_history.append({"role": "assistant", "content": complete_response})
            
            # Save session
            self._save_session()
            
            return complete_response
            
        except Exception as e:
            print(f"Error during research: {e}")
            return None
    
    def _save_session(self):
        """Save the current research session to a file"""
        try:
            if not os.path.exists(self.session_dir):
                os.makedirs(self.session_dir)
                
            filename = f"{self.session_dir}/session_{self.session_id}.json"
            with open(filename, "w") as f:
                json.dump({
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "history": self.session_history
                }, f, indent=2)
        except Exception as e:
            print(f"Failed to save session: {e}")
    
    def clear_session(self):
        """Clear the current research session"""
        self.session_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("Research session cleared. Starting fresh.")
    
    def focused_research(self, topic, depth=3, follow_up_strategy="breadth"):
        """Conduct deep research on a topic with automated follow-up questions"""
        print(f"Beginning focused research on: {topic}")
        
        # Initial research
        initial_response = self.research(topic, stream=True)
        
        if depth <= 1:
            return
        
        # Generate follow-up questions based on strategy
        if follow_up_strategy == "breadth":
            follow_up_prompt = (
                "Based on the research so far, identify 3 key aspects of this topic "
                "that would benefit from further exploration. Format as numbered questions."
            )
        elif follow_up_strategy == "depth":
            follow_up_prompt = (
                "Identify the most important or complex aspect of this topic that "
                "requires deeper investigation. Frame this as a specific question."
            )
        else:
            follow_up_prompt = (
                "What's the most interesting follow-up question to continue this research?"
            )
        
        # Get follow-up questions
        follow_up_questions_raw = self.research(follow_up_prompt, stream=False)
        
        # Parse questions (simple extraction)
        follow_up_questions = []
        for line in follow_up_questions_raw.split('\n'):
            if any(line.strip().startswith(str(i)) for i in range(1, 10)):
                follow_up_questions.append(line.strip())
        
        # If we didn't extract questions properly, just take the whole response
        if not follow_up_questions and follow_up_strategy == "depth":
            follow_up_questions = [follow_up_questions_raw]
        
        # Process follow-up questions recursively with reduced depth
        for i, question in enumerate(follow_up_questions[:min(3, len(follow_up_questions))]):
            print(f"\n--- Follow-up Research {i+1}/{len(follow_up_questions)} ---")
            print(f"Question: {question}\n")
            self.research(question, stream=True)
            
            # Only go deeper on the first question if we're in depth mode
            if follow_up_strategy == "depth" and i >= 0:
                break
                
            # Respect the maximum depth
            if depth > 2:
                # Further recursion with reduced depth
                next_strategy = follow_up_strategy  # maintain strategy
                self.focused_research(question, depth=depth-1, follow_up_strategy=next_strategy)


def main():
    parser = argparse.ArgumentParser(description="Advanced Research Assistant using Perplexity API")
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--api-key", help="Perplexity API key")
    parser.add_argument("--model", default="sonar-reasoning-pro", help="Model to use")
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for response generation")
    parser.add_argument("--focused", action="store_true", help="Perform focused research with follow-ups")
    parser.add_argument("--depth", type=int, default=2, help="Depth of focused research")
    parser.add_argument("--strategy", choices=["breadth", "depth"], default="breadth", help="Follow-up strategy")
    
    args = parser.parse_args()
    
    # Interactive mode if no query provided
    if not args.query:
        print("⚡ Research Assistant Activated ⚡")
        print("Type 'exit' to quit, 'clear' to start a new session")
        
        api_key = args.api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            api_key = input("Enter your Perplexity API key: ")
            
        assistant = ResearchAssistant(api_key=api_key)
        
        while True:
            query = input("\nWhat would you like to research? > ")
            
            if query.lower() == "exit":
                break
            elif query.lower() == "clear":
                assistant.clear_session()
                continue
                
            if args.focused:
                assistant.focused_research(query, depth=args.depth, follow_up_strategy=args.strategy)
            else:
                assistant.research(query, model=args.model, stream=args.stream, temperature=args.temperature)
    else:
        # One-off research mode
        assistant = ResearchAssistant(api_key=args.api_key)
        
        if args.focused:
            assistant.focused_research(args.query, depth=args.depth, follow_up_strategy=args.strategy)
        else:
            assistant.research(args.query, model=args.model, stream=args.stream, temperature=args.temperature)


if __name__ == "__main__":
    main()
