from sentence_transformers import SentenceTransformer
import chromadb
import torch
import tkinter as tk
from prompts import questions
import pyperclip
import csv
import time

promptNum = 0

def addRowCSV(file_path, prompt, answer):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['Prompt', 'Answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({'Prompt': prompt, 'Answer': answer})


def cudaDeviceSetup():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def intialSetup():
    global embedding_model
    global chroma_client
    global collection
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder = '/models/sentence-transformers/all-mpnet-base-v2')
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="RAG")

def get_overlapped_chunks(textin, chunksize, overlapsize):
    return [textin[a:a+chunksize] for a in range(0,len(textin), chunksize-overlapsize)]


def vectorDBSetup():
    
    chunks = get_overlapped_chunks(dataset, 1000, 100)
    chunk_embeddings = embedding_model.encode(chunks)
    chunk_embeddings.shape
    max_batch_size = 166  
    num_batches = (len(chunk_embeddings) + max_batch_size - 1) // max_batch_size
    for i in range(num_batches):
        start_idx = i * max_batch_size
        end_idx = (i + 1) * max_batch_size
        batch_embeddings = chunk_embeddings[start_idx:end_idx]
        batch_chunks = chunks[start_idx:end_idx]
        batch_ids = [str(j) for j in range(start_idx, min(end_idx, len(chunk_embeddings)))]
        
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_chunks,
            ids=batch_ids
        )


def retrieve_vector_db(query, n_results=15):
    results = collection.query(
    query_embeddings = embedding_model.encode(query).tolist(),
    n_results=n_results
    )
    return results['documents']


def getContext(prompt):
    query = prompt
    retrieved_results = retrieve_vector_db(query)
    context = ''
    for result in retrieved_results:
        context = '\n'.join(result)
    
    return context

def createContext():
    global contextPromptText
    global systemTrainPromptText
    context = getContext(setPrompt(promptNum))
    contextPromptText = f"""CONTEXT: {context}\QUESTION: {setPrompt(promptNum)}\nBased on the give context give answer or take relevant part from the context to form the answer. Keep the  answer simple and short paragraph."""
    contextPrpmpt.delete('1.0', tk.END)
    contextPrpmpt.insert('1.0', contextPromptText)
    systemTrainPromptText = f'''[INST] Give answer for the question  based on the context provided. Based on the give context give answer or take relevant part from the context to form the answer.Keep the  answer simple and short.Question: {setPrompt(promptNum)}Context : {context}[/INST]'''
    systemTrainPrompt.delete('1.0', tk.END)
    systemTrainPrompt.insert('1.0', systemTrainPromptText)

def greetAndSetupDialog(display):
    global dataset 
    try:
        dataset = open('dataset\\text-format\without-index\\Constitution.txt').read()
    except:
        print("File Error") 
   
    window = tk.Tk()
    greeting = tk.Label(
        window,
        text="Dataset Toolkit v0.1 By Aditya\nValidating Parameters... \nDONT CLOSE THIS WINDOW!!!",
        font=("Comic Sans", 25),       
        width=50,
        height=5,
        bg="black",
        fg="yellow")
    greeting.pack()
    print("Intial setup running....\nThis may take a while")
    cudaDeviceSetup()
    intialSetup()
    vectorDBSetup()
    print("Setup Complete!")
    window.after(display, window.destroy)  

def cudaNotAvailableWarningDialog(display):
    window = tk.Tk()
    greeting = tk.Label(
        window,
        font=("Arial", 25),
        text="Cuda Runtime not found!!\nToolkit may experience performance issues.",
        width=40,
        height=4,
       )
    greeting.pack()
    window.after(display, window.destroy)  

def setPrompt(index):
    return questions[index]

def changePrompt():
    global promptNum
    promptNum = (promptNum + 1) % len(questions)  
    promptEntry.delete('1.0', tk.END)
    promptEntry.insert('1.0', setPrompt(promptNum))
    promptNumLabel.config(text="Prompt No: "+str(promptNum + 1))

def copyText():
     pyperclip.copy(contextPromptText)

def pasteResult():
    global retrievedResult
    retrievedResult = pyperclip.paste()
    resultEntry.delete('1.0', tk.END)
    resultEntry.insert('1.0', retrievedResult)

def resetState():
    retrievedResult = ""
    contextPromptText = ""
    systemTrainPromptText = ""
    resultEntry.delete('1.0', tk.END)
    contextPrpmpt.delete('1.0', tk.END)
    systemTrainPrompt.delete('1.0', tk.END)

def exportToCSV():
  
    if not systemTrainPromptText:
        print(f"Error: Data can't be empty! Empty System Prompt")
    elif not retrievedResult:
        print(f"Error: Data can't be empty! Empty Result")
    else:
        addRowCSV("DatasetForFineTuning-Set1.csv", systemTrainPromptText, retrievedResult)
        print("Added Record successfully!")

startTime = time.time()
greetAndSetupDialog(5000)
endTime = time.time()

print("Execution time:",  endTime - startTime, "seconds")
if device.type == "cpu":
  cudaNotAvailableWarningDialog(6000)

mainWindow = tk.Tk()
mainWindow.geometry("1024x768")
mainWindow.title("Dataset Toolkit v0.1 By Aditya")
prompt = setPrompt(promptNum)
promptEntry = tk.Text(mainWindow, font=('calibre', 10, 'normal'), width=60, height=5)
promptEntry.insert('1.0', prompt)
promptEntry.pack()
promptEntry.place(x=10, y=10)
promptNumLabel = tk.Label(
        mainWindow,
        font=("Comic Sans", 10),
        text="Prompt No: 1")
promptNumLabel.place(x=10, y=100)

button1 = tk.Button(mainWindow, text="Next Prompt", command=changePrompt)
button1.place(x=10, y=150)

button2 = tk.Button(mainWindow, text="Create Context", command=createContext)
button2.place(x=100, y=150)

contextPrpmpt = tk.Text(mainWindow, font=('calibre', 10, 'normal'), width=60, height=30)
contextPrpmpt.place(x=10, y=200)


button3 = tk.Button(mainWindow, text="Copy to clipboard", command=copyText)
button3.place(x=10, y=700)

systemTrainPrompt = tk.Text(mainWindow, font=('calibre', 10, 'normal'), width=60, height=15)
systemTrainPrompt.place(x=500, y=10)

button4 = tk.Button(mainWindow, text="Paste Results", command=pasteResult)
button4.place(x=500, y=270)

resultEntry = tk.Text(mainWindow, font=('calibre', 10, 'normal'), width=60, height=20)
resultEntry.place(x=500, y=310)

button5 = tk.Button(mainWindow, text="Export to CSV", command=exportToCSV)
button5.place(x=500, y=650)

button5 = tk.Button(mainWindow, text="Reset State", command=resetState)
button5.place(x=600, y=650)


tk.mainloop()