# StudyBuddy
StudyBuddy is an AI study assistant that lets you upload a PDF and ask questions about it. Built with LangChain, Chroma, HuggingFace embeddings and Groq's LLM, it retrieves relevant passages and answers only from your document, replying "Sorry, answer not found" when the answer isn't there.


## Features

- Upload any text-based PDF from the sidebar
- Ask questions in a chat interface
- Answers are grounded in the document only, with no outside knowledge
- Replies **"Sorry, answer not found"** instead of guessing
- Chat history per session, with a one-click clear button

## How it works

1. **Load:** the PDF is read page by page with `PyPDFLoader`.
2. **Split:** the text is cut into overlapping chunks (1000 characters, 200 overlap).
3. **Embed:** each chunk is converted to a vector using `all-MiniLM-L6-v2`.
4. **Store:** vectors are kept in an in-memory Chroma database.
5. **Retrieve:** for each question, an MMR retriever picks the 4 most relevant, diverse chunks (`k=4`, `fetch_k=10`).
6. **Answer:** the chunks and the question go to Groq's `openai/gpt-oss-20b`, which is instructed to answer only from that context.
