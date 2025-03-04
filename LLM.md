# LLMs

[Good overview video by Andrej Karpathy](https://www.youtube.com/watch?v=7xTGNNLPyMI)

- Base Model: token predictor (no assistant capabilities)
- Instruct Model: assistent capabilities to ask questions about the data (text)
  - Smaller and faster trained than Base Model
  - Training set data is a series of Questions and Answers (labels written by humans)
  - See [Example](https://huggingface.co/datasets/allenai/olmo-2-hard-coded) of conversation dataset
  - You can use LLMs to ask what a good set of questions on a block of text or information would be and use that as the base for creating question/answer datasets

### Reducing Hallucination

- Need to find out what the model does not know, so that it doesn't make up a statistically generated answer
- You can use an LLM (Judge) to compare the answer another LLM gives to a question and determine whether the answer is correct
  - You should generate 3-5 answers from the other LLM (or however many you want) and compare all of them

### Computation per token

- Each token has a finite amount of computation, so ideally you want to spread out the computation needed across many tokens.
- Example: "The answer is 3. This is because we take the difference of..." (this provides the answer up front and puts all the computation needed into one token: 3).
  - It's better to break down the problem and get to the answer in smaller steps so that the computation per step (or token, etc.) is spread out.
  - Ask models to calculate intermediate results and steps, not just give the direct final answer

### LLMS are bad at Math

- If asking LLMs to run arithmetic or math, add at the end of the prompt "Use code."
  - This is a tool that LLMs like ChatGPT has to run a python interpreter to do the math so you can check it more easily.
- LLMs are bad at counting. Add the prompt "Use code." or tools available so the LLM can use something like a python interpreter to count a number of characters or items.
- Models are also bad at spelling or character level tasks. They see tokens (groups of characters), not individual letters

## Learning stages

### Pre-training

- Gather all the data, tokenized, probabilities worked out, etc.

### Post-training (Supervised Fine-tuning)

- Humans label answers to problems and questions

### Reinforcement Learning

- The model does practice problems
- A question or problem with the right answer is given to the model and it runs many solutions in parallel to see which paths generate the correct answer
- The best path is selected and paths that lead to an incorrect answer are discarded. Parameters/weights are adjusted to favor the best paths to make them more probable for types of questions/problems
- LLMs are capable enough to function as judges so you can have them look at the solution given by another LLM to determine if it matches the provided correct answer

### RLHF

- Reinforcement learning with a little human involvement for problems that are unverifiable
- Used for reinforcement learning of harder problems like producing jokes and humor, summarizing text, etc.
- Humans score a relatively low number of LLM outputs - usually people are asked to order the responses from best to worst (i.e. if the output is a generated joke)
- Another LLM is trained on the scores people gave - A "reward model"
- This reward model functions as a scorer simulator and can then score many more solutions given from the other LLM
- **IMPORTANT**: There is a diminishing returns using RLHF and you cannot let it go on forever like genuine Reinforcement Learning. After a number of passes, the results begin to degrade rapidly and you should not continue to use it past that point.

### Deep Seek

- Reinforcement learning revealed emergent property of better solutioning using "thinking" or "chain of thought" where the model re-traces it's steps and thinks through a problem
- Together.ai hosts the DeepSeek model online for use

## LLM Resources/Model Access

- [Together.ai](https://api.together.ai/)
- [Chat GPT](https://chatgpt.com)
- [Google Gemeni](https://gemini.google.com/app)
- [Hyperbolic - Base Models](https://app.hyperbolic.xyz/)
- [ChatGPT](chatgpt.com)
- [LMStudio](https://lmstudio.ai/) - app for running models locally