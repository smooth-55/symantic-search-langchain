from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from typing import TypedDict, Annotated, Optional, Literal
from constants import REVIEW_PROMPT, Datasets as documents, THRESHOLD, Transcript as user_transcript,PROMPT_TO_SPLIT_INTO_CHUNKS
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load variables from .env file
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ==================================================================================================


class Transcript(TypedDict):
    segments: Annotated[
        list[str],
        "List of very short micro-sentences created by aggressively splitting the transcript. The list should be long."
    ]

class Review(TypedDict):
    review: Literal['positive', 'negative'] 

class SymenticSearch:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.chat_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)
        self.prompt = PromptTemplate(
            template=f"{PROMPT_TO_SPLIT_INTO_CHUNKS} \n {{transcript}}",
            input_variables=["transcript"],
        )
        self.reviewPrompt = PromptTemplate(
            template=f"{REVIEW_PROMPT}",
            input_variables=["segment", "relevent_doc"],
        )
       
    

    def main(self):
        vector_docs = self.embeddings.embed_documents(documents)
        structured_model = self.chat_llm.with_structured_output(Transcript)
        structured_model2 = self.chat_llm.with_structured_output(Review)
        chain1 = self.prompt | structured_model 
        result = chain1.invoke({"transcript": user_transcript})
        segments = result["segments"]
        print(f"Total segments: {len(segments)}")
        # 4. Loop through each segment and search your database
        matched_task = set()
        output = list()
        okay = list()
        print(f"Total Task: {len(documents)}")
        for segment in segments:
            # Embed the specific segment
            segment_vector = self.embeddings.embed_query(segment)
            # similiarity scores with every segements
            similarity_scores = cosine_similarity([segment_vector], vector_docs)[0]

            for index, similarity_score in enumerate(similarity_scores):
                if similarity_score > THRESHOLD:
                    relevent_doc = documents[index]
                    if relevent_doc not in matched_task:
                        matched_task.add(relevent_doc)
                        # output.append({
                        #     "matchedSegment": segment,
                        #     "matchedWith": relevent_doc,
                        #     "confidence": similarity_score,
                        # })
                        chain2 = self.reviewPrompt | structured_model2
                        result = chain2.invoke({
                            "segment": segment,
                            "relevent_doc": relevent_doc,
                        })
                        print(f"Result from review: {type(result)}, {result}")
                        review = result["review"]
                        print(f"Review: {review} on the segment: {segment} with the matched doc: {relevent_doc}")
                        if review == "positive":
                            okay.append((
                            segment,
                            relevent_doc,
                        ))
                            print(f"Segment: {segment}")
                            print(f"Matched with: {relevent_doc}")
                        print(f"   Score: {similarity_score:.2f}")
        return okay


obj = SymenticSearch().main()



print(obj, len(obj))