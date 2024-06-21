import pandas as pd
import os
from ollama import Client
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from aas_loader import process_aasx_file, save_aasx, fill_template
from tqdm import tqdm
import pickle
import gradio as gr
import shutil


model="llama3"
base_url="http://127.0.0.1:11434"


def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    document = ["".join(each.page_content) for each in documents]
    texts = ""
    for page in document:
        texts += page

    text_instructions = "extract all the technical specification data of a product from the pdf into a list of technical properties in a format of '<property name>, <value>, <unit>' without any explanation or any texts, such as 'Nominal_Voltage, 250, V'. For a missing value, use 'N/A'. Each row must only contain these three elements with two ','. "
    
    client = Client(host=base_url)

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": text_instructions},
            {"role": "user", "content": texts}
        ],
        options={
        "temperature": 0
        }
    )
    technical_data_list = response['message']['content']

    def adjust_line(parts):
        return parts[:3] + ["N/A"] * (3 - len(parts))  # Adjust list to have exactly 3 elements

    lines = technical_data_list.strip().split("\n")
    data = [adjust_line(line.split(", ")) for line in lines]
    df = pd.DataFrame(data, columns=["Name", "Value", "Units"])
    return df


## semantic search
def LLM_matcher(query, candidates):
    system_prompt = "You will read two sentence-like entities to be matched. Each entity has several attributes such as descriptions.Your task is to decide whether the two entities are matched (they refer to the same entity). Only answer 'yes' or 'no'."
    CoT_prompt = "Think step by step. First, entities may be professional terminologies in specific domains, you should consider the domain knowledge. Second, the entity names and descriptions or definitions are most important. should consider synonyms. Third, entity1 may be defined for a specific use case or domain, while entity2 may be defined in more general terms. If the scope of Entity1 belongs to that of Entity2, they should be considered matching. However, if the scope of Entity2 belongs to that of Entity1, they should be considered not matching. Below are several examples"

    client = Client(host=base_url)

    for candidate in candidates:
        response = client.chat(
            model=model, # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": CoT_prompt},
                {"role": "user", "content": f"entity1: {query}\nentity2: {candidate}"}
            ],
            options={
        "temperature": 1
        }
        )
        print(f"\nentity1: {query}\nentity2: {candidate}")
        result = response['message']['content']
        if 'yes' in result:
            print("LLM Matching Result: ", candidate)
            return candidate
    print("LLM Matching Result: None")
    return 'None'

def process_files(pdf_file, aasx_file):
    # Save the uploaded files to process
    pdf_path = pdf_file.name
    aasx_path = aasx_file.name
    print(pdf_path)
    print(aasx_path)
    base_path = os.path.dirname(pdf_path)
    pdf_name = pdf_file.name.split("\\")[-1].split(".")[0]
    aas_smt = aasx_file.name.split("\\")[-1].split(".")[0]

    try:
        shutil.copy(aasx_path, os.path.join(base_path, aas_smt+'.aasx'))
    except shutil.SameFileError:
        pass
    except Exception as e:
        print("Error copying AASX file: ", e)
        return None
    aasx_path = os.path.join(base_path, aas_smt+'.aasx')


    # Process PDF
    if os.path.exists(os.path.join(base_path, pdf_name + '.csv')):
        pdf_df = pd.read_csv(os.path.join(base_path, pdf_name + '.csv'))
    else:
        pdf_df = process_pdf(pdf_path)
        pdf_df.to_csv(os.path.join(base_path, pdf_name + '.csv'), index=False)

    # Process AASX
    df_aasx, dict_aasx = process_aasx_file(aasx_path)
    df_aasx = df_aasx.drop('Index', axis=1)

    # corpus of eclass properties
    corpus = []
    for index, row in df_aasx.iterrows():
        corpus.append('name: ' + str(row['idShort']) + '.  description: ' + str(row['description']))

    # queries of aas properties
    queries = []
    for index, row in pdf_df.iterrows():
        queries.append('name: ' + str(row['Name']) + '.  unit: ' + str(row['Units']))

    # create embeddings for eclass and aas properties
    queries_embeddings_file = os.path.join(base_path, pdf_name + '_query.pkl')
    if os.path.exists(queries_embeddings_file):
        with open(queries_embeddings_file, 'rb') as f:
            queries_embeddings = pickle.load(f)
        print("Loaded precomputed queries_embeddings.")
    else:
        ollamaEmbed = OllamaEmbeddings(base_url=base_url, model=model)

        queries_embeddings = ollamaEmbed.embed_documents(queries)

        with open(queries_embeddings_file, 'wb') as f:
            pickle.dump(queries_embeddings, f)
        print("Saved new queries_embeddings.")

    aas_embeddings_file = os.path.join(base_path, aas_smt + '_corpus.pkl')
    if os.path.exists(aas_embeddings_file):
        with open(aas_embeddings_file, 'rb') as f:
            corpus_embeddings = pickle.load(f)
        print("Loaded precomputed eclass_embeddings.")
    else:
        ollamaEmbed = OllamaEmbeddings(base_url=base_url, model=model)

        corpus_embeddings = ollamaEmbed.embed_documents(queries)

        with open(aas_embeddings_file, 'wb') as f:
            pickle.dump(corpus_embeddings, f)
        print("Saved new corpus_embeddings.")

    result_temp_path = os.path.join(base_path, 'results_search.txt')
    if os.path.exists(result_temp_path):
        with open(result_temp_path, 'r') as f:
            results = f.readlines()
        print("Loaded precomputed match results.")
    else:
        from sentence_transformers import util
        import torch
        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = 10
        results = []
        save_epoch = 5
        count = 0
        threshold = 0.85
        for query_embedding in tqdm(queries_embeddings):
            # cosine-similarity and torch.topk to find the highest top_k scores
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            print("\n\n======================\n\n")
            print("Query:", queries[count])
            print("\nTop K most similar sentences in corpus:")

            candidates = []
            for score, idx in zip(top_results[0], top_results[1]):
                print(df_aasx['idShort'][int(idx)], "(Score: {:.4f})".format(score))
                if score > threshold:
                    candidates.append(corpus[int(idx)])
            if candidates is not None:
                candidate = LLM_matcher(queries[count], candidates)
            if candidate != 'None':
                results.append(
                    queries[count].split(".")[0].split(': ')[1] + ": " + candidate.split(".")[0].split(': ')[1])
            else:
                results.append(queries[count].split(".")[0].split(': ')[1] + ": None")
            count = count + 1

        # save the temporary results
        with open(result_temp_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write("%s\n" % item)

    result_dict = {}
    for result in results:
        key, value = result.split(": ")
        result_dict[key] = value
    match_results_df = pd.DataFrame(list(result_dict.items()), columns=["Manufacturer's Property", "SMT's Property matched by LLM"])

    fill_template(df_aasx, results, pdf_df, dict_aasx)
    output_aasx_path = aasx_path.replace(".aasx", "_filled.aasx")
    save_aasx(aasx_path, output_aasx_path, dict_aasx)

    # Clean up or provide download links
    return pdf_df, match_results_df, f"Processed AASX saved to {output_aasx_path} and PDF data processed.", output_aasx_path

# Create the Gradio app
iface = gr.Interface(
    fn=process_files,
    inputs=[
        gr.File(label="Upload PDF File"),
        gr.File(label="Upload AASX File")
    ],
    outputs=[
        gr.Dataframe(label="Extracted Data"),
        gr.Dataframe(label="Matched Results"),
        gr.Textbox(label="Status Message"),
        gr.File(label="Download Output File")
    ],
    title="LLM for AASX Populating with PDF Data",
    description="Upload a PDF and a AASX file to process them."
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()

