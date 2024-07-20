from quart import Quart, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Quart(__name__)

# モデルとトークナイザーを読み込む
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route('/ask', methods=['POST'])
async def ask():
    data = await request.get_json()
    input_text = data['question']
    
    inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    reply_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    
    return jsonify({'response': reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
