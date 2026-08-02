# Who's That Pokémon?

Professor Onas has called Asavari for her trainer certification exam tomorrow and the initial round is based on the classic Who's That Pokémon? where she'll
have to make a guess within 5 seconds! She's befuddled and cannot wrap her head around memorizing all the 151 pokémons in her pokédex. What she loves to do however, 
is to automate anything and everything, and she does exactly that. She built a model which can easily identify the pokémon, no matter what you throw at it. She
was fully confident that there's nothing which could stop her from acing the exam. 

Except, word of her plan somehow reached Team Rocket and them being their
notorious selves, introduced bugs in her program, without her noticing. You need to help Asavari out! Some bugs might be too obvious, easily detectable on the first glance but you need to be careful, some changes are subtle enough that the model may look like it's working fine, while actually underperforming badly. You've gotta catch 'em all!

![Who's That Pokemon banner](assets/banner.gif)

## How it works

This repository is organized as a module containing the end-to-end pipeline for the project. The project consists of two models working together.
1) A **U-Net** which segments the Pokémon out of its background, producing the silhouette mask.
2) A small **CNN classifier** which looks at that silhouette and names the  Pokémon.

See `whos_that_pokemon/models.py` for both architectures, and `main.py` for how they're wired together. 

## Project Structure

```
whos_that_pokemon/
├── data_prep.py          
├── datasets.py            
├── models.py               
├── losses.py                
├── train_segmentation.py
├── train_classifier.py
├── pipeline.py              
├── evaluate.py               
└── visualize.py              
main.py                        
```

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv), a Python project manager.

**Install `uv`**:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Get the project running:**

```bash
git clone https://github.com/anushka-priya/whos-that-pokemon.git
cd whos-that-pokemon
uv sync
uv run main.py
```
## Found a Bug?

Head over to the [Issues tab](../../issues). That's where the bugs Team Rocket left behind are tracked, one per issue. Pick one, and check
[CONTRIBUTING.md](CONTRIBUTING.md) for how to submit a fix.








