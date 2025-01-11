import json
import pandas as pd

def process_jokes_for_llama(input_file):
    """
    Process JSONL file to create training data for LLaMA model.
    Handles null values and invalid entries.
    """
    jokes = []
    skipped = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                joke_data = json.loads(line)
                
                # Skip invalid entries
                if not joke_data.get('selftext') or joke_data['selftext'] in ['[deleted]', '[removed]', 'null', '']:
                    skipped += 1
                    continue
                    
                # Create a structured prompt-completion pair
                joke_entry = {
                    'prompt': f"Generate a music-related joke with the following title: {joke_data.get('title', '')}\n\nJoke:",
                    'completion': str(joke_data['selftext']).strip(),
                    'metadata': {
                        'score': joke_data.get('score', 0),
                        'created_utc': joke_data.get('created_utc', ''),
                        'id': joke_data.get('id', '')
                    }
                }
                jokes.append(joke_entry)
            except Exception as e:
                print(f"Error processing line {line_number}: {str(e)}")
                skipped += 1
                continue
    
    print(f"Processed {len(jokes)} valid jokes, skipped {skipped} invalid entries")
    return jokes

def format_for_llama(jokes):
    """
    Format jokes into LLaMA training format.
    """
    formatted_data = []
    
    for joke in jokes:
        if joke['completion'] and joke['prompt']:  # Ensure both fields exist
            entry = {
                'instruction': 'Generate a funny music-related joke',
                'input': joke['prompt'],
                'output': joke['completion']
            }
            formatted_data.append(entry)
    
    return formatted_data

def save_training_data(formatted_data, output_file):
    """
    Save formatted data as JSONL for LLaMA training.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def analyze_dataset(jokes):
    """
    Analyze the dataset and print statistics
    """
    if not jokes:
        print("No valid jokes to analyze!")
        return
        
    df = pd.DataFrame(jokes)
    print("\nDataset Statistics:")
    print(f"Total unique jokes: {len(df)}")
    
    if 'metadata' in df.columns:
        scores = df['metadata'].apply(lambda x: x.get('score', 0))
        print(f"Average joke score: {scores.mean():.2f}")
        print(f"Highest score: {scores.max()}")
        print(f"Lowest score: {scores.min()}")

def main():
    input_file = '/Users/muhammad/Documents/project_cygnet/CYGNET/reddit_music_jokes.jsonl'

    output_file = '/Users/muhammad/Documents/project_cygnet/CYGNET/llama_training_data.jsonl'
    
    try:
        # Process jokes
        jokes = process_jokes_for_llama(input_file)
        
        if not jokes:
            print("No valid jokes were processed!")
            return
            
        # Format for LLaMA
        formatted_data = format_for_llama(jokes)
        
        # Save training data
        save_training_data(formatted_data, output_file)
        
        # Analyze dataset
        analyze_dataset(jokes)
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()