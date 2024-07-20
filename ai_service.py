from quart import Quart, request, jsonify
import openai
import os

app = Quart(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/ask', methods=['POST'])
async def ask():
    data = await request.get_json()
    question = data['question']
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=question,
        max_tokens=100
    )
    return jsonify({'response': response.choices[0].text.strip()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
