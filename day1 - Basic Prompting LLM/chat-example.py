from gemini_setup import client

chat = client.chats.create(model='gemini-2.0-flash', history=[])

response = chat.send_message('Hello! My name is Zlork.')
print(response.text)

response = chat.send_message('Can you tell me something interesting about dinosaurs?')
print(response.text)

response = chat.send_message('Do you remember what my name is?')
print(response.text)