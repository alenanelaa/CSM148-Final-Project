# JokeGPT

Using the framework provided by Dr. Andrej Karpathy (https://github.com/karpathy/nanoGPT), our team created a mini model with the aim of generating jokes. Our model was first trained on English language text, then various datasets of setup-punchline jokes scraped from primarily Reddit.com

## Interface Running Steps

1. Download the repository and open a terminal in the repository folder
2. Install gradio with `pip install gradio` and torch with `pip install torch`
3. Set `MODEL_PATH = "./your_model_name.pth"` in the `interface.py` file
4. If you want to modify the length of the output set `MIN_CHAR_OUTPUT` to the desired (integer) value.
5. Run the command `python3 interface.py` in the terminal
6. Connect to the local server provided in the output.

## Contributors

Hank Lin // hank0212 // lintsunghan2019@gmail.com

Sasan Esfahani // isfahanisasan // sesfahani@ucla.edu

Victor Chinnappan // vchinn04 // vchinn04@gmail.com

Alena Zhu // alenanelaa // alena.k.zhu@gmail.com