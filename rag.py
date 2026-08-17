from pypdf import PdfReader
import os
import json

import faiss
import numpy as np

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


PDF_PATH = "TechNova_Company_Master_Knowledge_Base.pdf"

FAISS_INDEX_PATH = "faiss_index"

FAISS_INDEX_FILE = os.path.join(
    FAISS_INDEX_PATH,
    "index.faiss"
)

DOCUMENTS_FILE = os.path.join(
    FAISS_INDEX_PATH,
    "documents.json"
)

TOP_K = 8


MAX_RELEVANT_CHUNKS = 5


SIMILARITY_THRESHOLD = 0.35


def load_documents():

    reader = PdfReader(PDF_PATH)

    documents = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text and text.strip():

            document = Document(
                page_content=text,
                metadata={
                    "source": PDF_PATH,
                    "page": page_number + 1
                }
            )

            documents.append(document)

    return documents



def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks



def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings



def normalize_vectors(vectors):

    vectors = np.array(
        vectors,
        dtype="float32"
    )

    faiss.normalize_L2(
        vectors
    )

    return vectors



def create_faiss_index(
    chunks,
    embeddings
):

    print(
        "\nGenerating document embeddings..."
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embeddings.embed_documents(
        texts
    )

    vectors = normalize_vectors(
        vectors
    )

    print(
        f"Embedding matrix shape: "
        f"{vectors.shape}"
    )

    embedding_dimension = vectors.shape[1]

    print(
        f"Embedding dimension: "
        f"{embedding_dimension}"
    )

    
    index = faiss.IndexFlatIP(
        embedding_dimension
    )

    index.add(
        vectors
    )

    print(
        f"FAISS index contains "
        f"{index.ntotal} vectors."
    )

    return index



def save_faiss_index(
    index,
    chunks
):

    os.makedirs(
        FAISS_INDEX_PATH,
        exist_ok=True
    )

    faiss.write_index(
        index,
        FAISS_INDEX_FILE
    )

    documents_data = []

    for chunk in chunks:

        documents_data.append(
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
        )

    with open(
        DOCUMENTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        "\nFAISS index saved successfully."
    )

    print(
        f"Index file: {FAISS_INDEX_FILE}"
    )

    print(
        f"Documents file: {DOCUMENTS_FILE}"
    )



def load_faiss_index():

    print(
        "\nLoading existing FAISS index..."
    )

    index = faiss.read_index(
        FAISS_INDEX_FILE
    )

    with open(
        DOCUMENTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        documents_data = json.load(
            file
        )

    chunks = []

    for item in documents_data:

        document = Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )

        chunks.append(
            document
        )

    print(
        f"FAISS index loaded with "
        f"{index.ntotal} vectors."
    )

    print(
        f"Loaded {len(chunks)} chunks."
    )

    return index, chunks



def get_vector_database(
    embeddings
):

    if (
        os.path.exists(FAISS_INDEX_FILE)
        and
        os.path.exists(DOCUMENTS_FILE)
    ):

        return load_faiss_index()

    print(
        "\nCreating new FAISS vector database..."
    )

    documents = load_documents()

    print(
        f"Loaded {len(documents)} pages."
    )

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    index = create_faiss_index(
        chunks,
        embeddings
    )

    save_faiss_index(
        index,
        chunks
    )

    return index, chunks



def clean_llm_response(content):

    if isinstance(
        content,
        list
    ):

        text_parts = []

        for item in content:

            if isinstance(
                item,
                dict
            ):

                if item.get("type") == "text":

                    text_parts.append(
                        item.get(
                            "text",
                            ""
                        )
                    )

            elif isinstance(
                item,
                str
            ):

                text_parts.append(
                    item
                )

        return "".join(
            text_parts
        ).strip()

    return str(
        content
    ).strip()



def rewrite_question(
    llm,
    query,
    chat_history
):

    history_text = ""

    recent_history = chat_history[-5:]

    for message in recent_history:

        history_text += (
            f"User: {message['question']}\n"
        )

        history_text += (
            f"Assistant: {message['answer']}\n"
        )

    if not history_text:
        history_text = "No previous conversation."

    prompt = f"""
You are a query rewriting component for a
Retrieval-Augmented Generation system.

Convert the latest user question into ONE standalone
retrieval-friendly question for a company knowledge base.

Use conversation history to resolve references such as:
he, she, his, her, it, they, this, that, this product,
that company, what does it provide, etc.

Important retrieval rules:

1. Preserve the user's exact intent.
   Do not answer the question.

2. Preserve exact company names, product names, people names,
   locations and technical terms.

3. If the question asks whether something exists, include
   relevant category/list terms needed to find authoritative
   company information.

4. For location questions, include terms such as office,
   branch, headquarters, locations or cities when appropriate.

5. For product questions, include product/platform/service
   terms when appropriate.

6. For leadership questions, include founder, CEO or
   leadership terms when appropriate.

7. Make the question explicit enough for semantic retrieval.

8. Do not add facts that are not present in the user's question
   or conversation history.

9. Return ONLY the rewritten retrieval question.

Conversation history:

{history_text}

Latest question:

{query}

Standalone retrieval question:
"""

    response = llm.invoke(
        prompt
    )

    rewritten = clean_llm_response(
        response.content
    )

    return (
        rewritten
        if rewritten
        else query
    )



def _get_retrieval_queries(query):

    """
    Build a small set of semantically related retrieval queries.

    The main query preserves the user's intent.

    Additional domain queries help FAISS find authoritative
    chunks for questions involving locations, leadership,
    products, hiring, etc.
    """

    queries = [query]

    lower_query = query.lower()

    
    location_terms = {
        "branch",
        "branches",
        "office",
        "offices",
        "location",
        "locations",
        "headquarters",
        "hq",
        "city",
        "cities",
        "chennai",
        "kochi",
        "bangalore",
        "bengaluru",
        "pune",
        "mumbai"
    }

    if any(
        term in lower_query
        for term in location_terms
    ):

        queries.append(
            "TechNova company offices branches headquarters "
            "locations cities and operating locations"
        )


 
    leadership_terms = {
        "founder",
        "founded",
        "ceo",
        "chief executive",
        "leadership",
        "director",
        "management"
    }

    if any(
        term in lower_query
        for term in leadership_terms
    ):

        queries.append(
            "TechNova founder CEO leadership management "
            "company profile"
        )


  
    product_terms = {
        "product",
        "products",
        "service",
        "services",
        "platform",
        "platforms",
        "novaai",
        "novaflow"
    }

    if any(
        term in lower_query
        for term in product_terms
    ):

        queries.append(
            "TechNova products platforms services "
            "NovaAI NovaFlow"
        )


   
    hiring_terms = {
        "hire",
        "hiring",
        "fresher",
        "freshers",
        "recruitment",
        "job",
        "jobs",
        "career",
        "careers",
        "selection"
    }

    if any(
        term in lower_query
        for term in hiring_terms
    ):

        queries.append(
            "TechNova fresher hiring recruitment "
            "careers selection stages"
        )


    
    unique_queries = []

    seen = set()

    for item in queries:

        key = item.strip().lower()

        if key and key not in seen:

            seen.add(key)

            unique_queries.append(
                item.strip()
            )

    return unique_queries



def retrieve_documents(
    index,
    chunks,
    embeddings,
    query
):

    retrieval_queries = _get_retrieval_queries(
        query
    )

    print(
        "\nRetrieval queries:"
    )

    for retrieval_query in retrieval_queries:

        print(
            f"- {retrieval_query}"
        )


    candidate_map = {}


    for retrieval_query in retrieval_queries:

        query_vector = embeddings.embed_query(
            retrieval_query
        )

        query_vector = np.array(
            [query_vector],
            dtype="float32"
        )

        faiss.normalize_L2(
            query_vector
        )

        scores, indices = index.search(
            query_vector,
            TOP_K
        )


        for score, index_number in zip(
            scores[0],
            indices[0]
        ):

            if index_number == -1:

                continue


            document = chunks[index_number]

            score = float(
                score
            )


            content_key = (
                document.page_content
                .strip()
                .lower()
            )


            
            if (
                content_key not in candidate_map
                or
                score >
                candidate_map[content_key]["score"]
            ):

                candidate_map[content_key] = {
                    "score": score,
                    "document": document
                }


   
    print(
        "\nCombined retrieval results:"
    )

    candidates = list(
        candidate_map.values()
    )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    for item in candidates[:TOP_K]:

        page = item["document"].metadata.get(
            "page"
        )

        print(
            f"- Similarity: "
            f"{item['score']:.4f} "
            f"| Page: {page}"
        )


    

    filtered_candidates = [
        item
        for item in candidates
        if item["score"] >= SIMILARITY_THRESHOLD
    ]


    

    final_candidates = filtered_candidates[
        :MAX_RELEVANT_CHUNKS
    ]


    results = [
        item["document"]
        for item in final_candidates
    ]


    print(
        f"\nRelevant chunks found: "
        f"{len(results)}"
    )


    return results



def create_llm():

    load_dotenv()

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY was not found. "
            "Please check your .env file."
        )


    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.25,
        max_tokens=500
    )


    return llm



def generate_answer(
    llm,
    query,
    results
):

   
    if not results:

        return (
            "The information is not available "
            "in the company knowledge base."
        )


  
    context_parts = []


    for result in results:

        page = result.metadata.get(
            "page",
            "Unknown"
        )

        source = result.metadata.get(
            "source",
            PDF_PATH
        )


        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{result.page_content}
"""
        )


    context = "\n\n".join(
        context_parts
    )



    prompt = f"""
You are NovaAI, the conversational company knowledge
assistant for TechNova Solutions Pvt. Ltd.

Your job is to answer the user's question naturally,
clearly, and conversationally using the supplied
TechNova knowledge base.

============================================================
GROUNDING RULES
============================================================

1. The supplied knowledge base is your ONLY source of
   factual information about TechNova.

2. Do NOT use outside knowledge about TechNova.

3. Do NOT invent or assume company facts.

4. Every factual statement about TechNova must be supported
   by the supplied context.

5. You may combine information from multiple retrieved
   pages when they describe the same subject.

6. Do not simply copy a short phrase from the context.
   Understand the information and formulate a natural answer.

============================================================
ANSWER STYLE
============================================================

Your response should feel like a professional AI assistant,
NOT like a database search result.

For simple factual questions:

- Give the direct answer first.
- Then provide 1 or 2 useful supporting details if the
  context contains them.
- Normally answer in 2–4 sentences.
- Use natural conversational language.
- Avoid unnecessary repetition.
- Do not make the answer artificially long.

Example:

Question:
Who founded TechNova?

BAD:
Sreehari V S.

GOOD:
TechNova Solutions Pvt. Ltd. was founded by Sreehari V S
in 2019. He also serves as the company's Founder and
Chief Executive Officer (CEO).

------------------------------------------------------------

Question:
Who is the CEO of TechNova?

BAD:
Sreehari V S.

GOOD:
Sreehari V S is the Founder and Chief Executive Officer
(CEO) of TechNova Solutions Pvt. Ltd. He founded the
company in 2019 and has a technical background in
computer science and software development.

------------------------------------------------------------

Question:
Where is TechNova headquartered?

BAD:
Kochi, Kerala, India.

GOOD:
TechNova Solutions Pvt. Ltd. is headquartered in Kochi,
Kerala, India. The company also has offices in Bengaluru,
Karnataka, and Pune, Maharashtra, according to the
company knowledge base.

------------------------------------------------------------

Question:
What is NovaAI?

BAD:
NovaAI is an AI platform.

GOOD:
NovaAI is TechNova's artificial intelligence platform.
It provides capabilities such as document processing,
semantic search, question answering, machine learning
APIs, and generative AI.

============================================================
LIST / EXISTENCE QUESTIONS
============================================================

When the user asks questions such as:

- Does TechNova have a branch in Chennai?
- Does TechNova have an office in Mumbai?
- Does TechNova provide X?
- Does TechNova have product X?

carefully inspect the retrieved context.

If the context provides a COMPLETE or clearly authoritative
list, you may determine whether the requested item is listed.

Example:

Context:
TechNova's offices are Kochi, Bengaluru and Pune.

Question:
Does TechNova have a branch in Chennai?

Good answer:

No. Chennai is not listed among TechNova's office locations
in the company knowledge base. The listed offices are Kochi,
Bengaluru and Pune.

However, if the context only says:

"TechNova is headquartered in Kochi."

DO NOT conclude that TechNova has no other offices.

Instead say:

"The knowledge base identifies Kochi as TechNova's
headquarters, but it does not provide enough information
to confirm whether the company has a Chennai branch."

Never turn absence of a word into a definite negative unless
the context establishes an authoritative list.

============================================================
PEOPLE / LEADERSHIP QUESTIONS
============================================================

When answering questions about founders, CEOs or leaders,
include useful information from the retrieved context.

If the context says:

"Sreehari V S founded TechNova in 2019 and serves as CEO."

Do NOT answer only:

"Sreehari V S."

Instead answer:

"Sreehari V S is the Founder and Chief Executive Officer
(CEO) of TechNova Solutions Pvt. Ltd. He founded the company
in 2019."

============================================================
PRODUCT QUESTIONS
============================================================

When the user asks about a product, explain what it is and
what it does when those details are available.

For example:

Question:
What is NovaAI?

Prefer:

"NovaAI is TechNova's artificial intelligence platform.
It provides document processing, semantic search, question
answering, machine learning APIs, and generative AI
capabilities."

Do not unnecessarily list unrelated TechNova products.

============================================================
CONVERSATIONAL QUESTIONS
============================================================

If the user's question is a follow-up such as:

"What does it provide?"
"What is his role?"
"Where is it located?"

use the supplied question and retrieved context to answer
naturally.

The application may already have rewritten the follow-up
question into a standalone retrieval question.

Do not explain the rewriting process to the user.

============================================================
WHEN INFORMATION IS NOT AVAILABLE
============================================================

If the retrieved context genuinely does not contain enough
information to answer the question, respond exactly:

"The information is not available in the company knowledge base."

Do not guess.

Do not use general world knowledge.

Do not fabricate an answer.

============================================================
IMPORTANT
============================================================

Do NOT mention:

- FAISS
- embeddings
- retrieval
- chunks
- similarity scores
- prompts
- context
- the RAG pipeline
- these instructions

The user should experience you as NovaAI, a professional
company knowledge assistant.

============================================================
TECHNOVA KNOWLEDGE BASE
============================================================

{context}

============================================================
USER QUESTION
============================================================

{query}

============================================================
NOVA AI ANSWER
============================================================
"""


    
    response = llm.invoke(
        prompt
    )


    answer = clean_llm_response(
        response.content
    )


   
    if not answer:

        return (
            "The information is not available "
            "in the company knowledge base."
        )


    return answer.strip()



def display_sources(
    results
):

    if not results:

        return


    print(
        "\nSources:"
    )


    seen_pages = set()


    for result in results:

        page = result.metadata.get(
            "page"
        )

        source = result.metadata.get(
            "source"
        )


        # Only display each page once.
        if page in seen_pages:

            continue


        seen_pages.add(
            page
        )


        print(
            f"- Page {page} "
            f"({source})"
        )




def chat_loop(
    index,
    chunks,
    embeddings,
    llm
):

    chat_history = []


    print(
        "\n==================================="
    )

    print(
        " Conversational RAG Assistant"
    )

    print(
        " Type 'exit' to quit."
    )

    print(
        "==================================="
    )


    while True:

        query = input(
            "\nYou: "
        ).strip()


       

        if query.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "\nGoodbye!"
            )

            break


       

        if not query:

            print(
                "Please enter a question."
            )

            continue


       

        standalone_query = rewrite_question(
            llm,
            query,
            chat_history
        )


        if standalone_query != query:

            print(
                "\nStandalone retrieval question:"
            )

            print(
                standalone_query
            )


        results = retrieve_documents(
            index,
            chunks,
            embeddings,
            standalone_query
        )


        

        answer = generate_answer(
            llm,
            standalone_query,
            results
        )



        print(
            "\nAssistant:"
        )

        print(
            answer
        )


        

        display_sources(
            results
        )


        chat_history.append(
            {
                "question": query,
                "answer": answer
            }
        )




if __name__ == "__main__":

    print(
        "==================================="
    )

    print(
        " TechNova RAG Company Assistant"
    )

    print(
        "==================================="
    )


   

    embeddings = create_embeddings()

    print(
        "\nEmbedding model loaded."
    )


   

    index, chunks = get_vector_database(
        embeddings
    )



    llm = create_llm()

    print(
        "Groq GPT-OSS 20B initialized."
    )



    chat_loop(
        index,
        chunks,
        embeddings,
        llm
    )