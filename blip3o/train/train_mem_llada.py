from blip3o.train.train_llada import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
