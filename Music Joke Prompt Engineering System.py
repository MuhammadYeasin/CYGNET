import json
import pandas as pd
import re
from datetime import datetime
import os
import ollama
from typing import List, Dict, Optional

# [Previous MusicJokePromptEngineer and JokeSaver classes remain the same]

class MusicJokePromptEngineer:
    def __init__(self, input_file: str):
        """Initialize prompt engineer with joke data file."""
        self.jokes = self.load_jokes(input_file)
        self.categories = {
            'puns': r'\b(like|sounds|called|difference|between)\b',
            'instrument': r'\b(guitar|piano|drum|violin|bass|trumpet|tuba)\b',
            'musician': r'\b(musician|band|singer|player|conductor)\b',
            'music_theory': r'\b(note|chord|scale|key|flat|sharp)\b'
        }
    
    def load_jokes(self, input_file: str) -> List[Dict]:
        """Load and process jokes from JSONL file."""
        jokes = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        joke = json.loads(line)
                        if joke.get('metadata') and joke.get('completion'):
                            jokes.append(joke)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Warning: Could not find {input_file}. Starting with empty joke list.")
        return jokes

    def generate_prompt(self, category: str = None, n_examples: int = 2) -> str:
        """Generate a structured prompt with examples."""
        top_jokes = self.get_top_jokes(category, n_examples)
        
        prompt_parts = [
            "Generate a funny music-related joke following these guidelines:",
            "",
            "1. Make sure the joke is music-themed",
            "2. Include a clear setup and punchline",
            "3. Use wordplay, puns, or musical terminology creatively",
            ""
        ]
        
        if category:
            prompt_parts.extend([
                f"Focus on {category}-related jokes.",
                ""
            ])
        
        if top_jokes:
            prompt_parts.append("Here are some examples of good music jokes:\n")
            
            for i, joke in enumerate(top_jokes, 1):
                title = joke['prompt'].split(": ")[-1].strip()
                punchline = joke['completion'].strip()
                prompt_parts.extend([
                    f"Example {i}:",
                    f"Title: {title}",
                    f"Punchline: {punchline}",
                    ""
                ])
        
        prompt_parts.extend([
            "Now, generate a new music joke following a similar style.",
            "Make it original and clever.",
            "",
            "Your joke:"
        ])
        
        return "\n".join(prompt_parts)

    def get_top_jokes(self, category: str = None, n: int = 3) -> List[Dict]:
        """Get top-scoring jokes, optionally filtered by category."""
        filtered_jokes = self.jokes
        if category:
            filtered_jokes = [
                joke for joke in self.jokes 
                if category in self.categorize_joke(joke)
            ]
        
        return sorted(
            filtered_jokes,
            key=lambda x: x['metadata'].get('score', 0),
            reverse=True
        )[:n]

    def categorize_joke(self, joke: Dict) -> List[str]:
        """Determine joke categories based on content."""
        categories = []
        combined_text = f"{joke['prompt']} {joke['completion']}".lower()
        
        for category, pattern in self.categories.items():
            if re.search(pattern, combined_text):
                categories.append(category)
        return categories

class JokeSaver:
    def __init__(self, output_file: str = "generated_music_jokes.jsonl"):
        self.output_file = output_file
        
    def save_joke(self, joke_data: Dict):
        """Save a single joke to the JSONL file."""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            json.dump(joke_data, f, ensure_ascii=False)
            f.write('\n')
    
    def save_batch(self, jokes: List[Dict]):
        """Save multiple jokes to the JSONL file."""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            for joke in jokes:
                json.dump(joke, f, ensure_ascii=False)
                f.write('\n')


class MusicJokeSystem:
    def __init__(self, model_name: str = 'llama3.2', training_data_path: str = '/Users/muhammad/Documents/project_cygnet/CYGNET/llama_training_data.jsonl', 
                 output_file: str = "/Users/muhammad/Documents/project_cygnet/CYGNET/generated_music_jokes.jsonl"):
        """Initialize the joke generation system using Ollama."""
        self.prompt_engineer = MusicJokePromptEngineer(training_data_path)
        self.joke_saver = JokeSaver(output_file)
        self.model_name = model_name

    def generate_joke(self, category: Optional[str] = None, temperature: float = 0.7) -> Dict:
        """Generate a joke using Ollama and save it."""
        prompt = self.prompt_engineer.generate_prompt(category)
        
        try:
            # Generate using Ollama with correct parameter structure
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': temperature,
                    'num_predict': 100,  # Adjust this for response length
                    'top_p': 0.9
                }
            )
            
            # Extract the generated joke text
            joke_text = response['message']['content'].strip()
            
            # Prepare joke data for saving
            joke_data = {
                'timestamp': datetime.now().isoformat(),
                'category': category if category else 'general',
                'prompt': prompt,
                'generated_joke': joke_text,
                'temperature': temperature,
                'model': self.model_name
            }
            
            # Save the joke
            self.joke_saver.save_joke(joke_data)
            
            return joke_data
            
        except Exception as e:
            print(f"Error generating joke: {str(e)}")
            # Print the full error for debugging
            import traceback
            print(traceback.format_exc())
            return None

    def generate_batch(self, n_jokes_per_category: int = 3, temperature: float = 0.7) -> List[Dict]:
        """Generate multiple jokes for each category."""
        all_jokes = []
        categories = list(self.prompt_engineer.categories.keys()) + [None]  # Include general category
        
        for category in categories:
            print(f"\nGenerating {n_jokes_per_category} jokes for category: {category if category else 'general'}")
            for i in range(n_jokes_per_category):
                joke_data = self.generate_joke(category, temperature)
                if joke_data:
                    all_jokes.append(joke_data)
                    print(f"Generated joke {i+1}/{n_jokes_per_category}:")
                    print(joke_data['generated_joke'])
                    print("-" * 50)
        
        return all_jokes

def main():
    # Initialize the system
    joke_system = MusicJokeSystem(model_name='llama2')
    
    try:
        # Generate batch of jokes
        print("\nGenerating jokes...")
        generated_jokes = joke_system.generate_batch(n_jokes_per_category=2, temperature=0.8)
        
        # Print summary
        print(f"\nGenerated {len(generated_jokes)} jokes total")
        
        # Print all generated jokes
        print("\nGenerated Jokes:")
        for i, joke in enumerate(generated_jokes, 1):
            print(f"\n{i}. Category: {joke['category']}")
            print(f"Joke: {joke['generated_joke']}")
            print("-" * 50)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()