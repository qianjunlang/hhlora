import numpy as np
from torch import nn, optim

from utils import model,tokenized,tokenizer,greedy_gen,untokenize,soft_p_list





train_text = "it is a truth universally acknowledged, that a single"
train_input_ids = tokenized(train_text)
labels = train_input_ids.clone()

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(3):
    optimizer.zero_grad()

    outputs = model(input_ids=train_input_ids, labels=labels)
    loss = outputs.loss

    loss.backward()

    optimizer.step()

    print(f"Training loss: {loss.item()}")




#-=-=-=-=-=

model.eval()  # with torch.no_grad():

test_text = "it is a truth universally acknowledged,"

output = greedy_gen(tokenized(test_text))

print(untokenize(output))