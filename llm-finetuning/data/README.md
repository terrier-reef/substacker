# Training Data

Add your creative writing samples here in JSONL format.

## Format

Each line is a JSON object with a `text` field:

```jsonl
{"text": "Your story here..."}
{"text": "Another story..."}
```

## Example

```jsonl
{"text": "Once upon a time, in a kingdom hidden behind mountains of mist, there lived a young adventurer named Aria. She dreamed of exploring the world beyond her village, of discovering ancient ruins and lost civilizations. One day, a mysterious map arrived at her doorstep..."}
{"text": "The old bookstore smelled of leather and forgotten dreams. Every shelf held worlds waiting to be discovered. Maya spent countless afternoons there, lost in stories that made her forget about the real world entirely. But one afternoon, she found a book that seemed to be about her own life..."}
```

## Tips for Better Results

### 1. Quality Stories
- Write well-developed narratives
- Include vivid descriptions
- Show character emotions and motivations
- Create engaging plots

### 2. Consistent Style
Fine-tune on stories with similar:
- Tone (whimsical, dark, romantic, etc.)
- Genre (fantasy, sci-fi, mystery, etc.)
- Length (~500-1500 words each)

### 3. Minimum Data
- **For testing**: 5-10 stories
- **For real fine-tuning**: 20-50 stories
- **For best results**: 100+ stories

### 4. Diversity
Mix different themes:
- Fantasy adventures
- Intimate character moments
- Action and suspense
- Wonder and discovery

## Creating Data Programmatically

```python
import json

stories = [
    "Story 1...",
    "Story 2...",
    "Story 3...",
]

# Write to JSONL
with open("creative_stories.jsonl", "w") as f:
    for story in stories:
        json.dump({"text": story}, f)
        f.write("\n")
```

## Common Formats to Convert

### From text file (one story per file)

```python
import json
import os

output = []
for filename in os.listdir("my_stories"):
    if filename.endswith(".txt"):
        with open(f"my_stories/{filename}") as f:
            text = f.read()
        output.append({"text": text})

with open("creative_stories.jsonl", "w") as f:
    for item in output:
        f.write(json.dumps(item) + "\n")
```

### From markdown file (split by headers)

```python
import json

with open("stories.md") as f:
    content = f.read()

stories = content.split("## ")  # Split by story headers

output = []
for story in stories:
    if story.strip():
        output.append({"text": story.strip()})

with open("creative_stories.jsonl", "w") as f:
    for item in output:
        f.write(json.dumps(item) + "\n")
```

## Validation

Check your JSONL file:

```python
import json

with open("creative_stories.jsonl") as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            if "text" not in data:
                print(f"Line {i}: Missing 'text' field")
            elif len(data["text"]) < 100:
                print(f"Line {i}: Story too short ({len(data['text'])} chars)")
        except json.JSONDecodeError:
            print(f"Line {i}: Invalid JSON")

print("✓ Validation complete")
```

## Data Size

- **5 stories** (~3000 words): 30 KB, trains in ~30 min
- **20 stories** (~12000 words): 120 KB, trains in ~1 hour
- **50 stories** (~30000 words): 300 KB, trains in ~2 hours
- **100 stories** (~60000 words): 600 KB, trains in ~3 hours

CPU training time scales roughly linearly with data size.

## Getting More Data

### Free Resources
- [Project Gutenberg](https://www.gutenberg.org/) - Public domain books
- [AI Story Collection](https://archive.org) - Archived stories
- [Reddit Writing Prompts](https://reddit.com/r/WritingPrompts/) - Creative stories
- [NaNoWriMo](https://nanowrimo.org) - Writer community

### Custom Data
Best results come from **your own writing**:
- Paste your finished stories
- Include your writing style and preferences
- Fine-tune creates a model that writes like you!

## Next Steps

1. Add stories to `creative_stories.jsonl`
2. Run `python train.py`
3. Load fine-tuned model in LM Studio
4. Generate new creative content!
