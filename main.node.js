import "dotenv/config";
import {
  GoogleGenerativeAIEmbeddings,
  ChatGoogleGenerativeAI,
} from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import cosineSimilarity from "compute-cosine-similarity";
import {
  Datasets as documents,
  THRESHOLD,
  Transcript as userTranscript,
  PROMPT_TO_SPLIT_INTO_CHUNKS,
  REVIEW_PROMPT,
} from "./constants.node.js";

const TranscriptSchema = z.object({
  segments: z.array(z.string()),
});

const ReviewSchema = z.object({
  review: z.enum(["positive", "negative"]),
});

class SemanticSearch {
  constructor() {
    this.embeddings = new GoogleGenerativeAIEmbeddings({
      model: "gemini-embedding-001",
    });

    this.chatLlm = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash",
      temperature: 0,
    });

    this.prompt = PromptTemplate.fromTemplate(
      `${PROMPT_TO_SPLIT_INTO_CHUNKS}\n{transcript}`
    );

    this.reviewPrompt = PromptTemplate.fromTemplate(REVIEW_PROMPT);
  }

  async run() {
    const vectorDocs = await this.embeddings.embedDocuments(documents);

    const transcriptChain = this.prompt.pipe(
      this.chatLlm.withStructuredOutput(TranscriptSchema)
    );

    const reviewChain = this.reviewPrompt.pipe(
      this.chatLlm.withStructuredOutput(ReviewSchema)
    );

    const transcriptResult = await transcriptChain.invoke({
      transcript: userTranscript,
    });

    const segments = transcriptResult.segments;
    console.log(`Total segments: ${segments.length}`);
    console.log(`Total Task: ${documents.length}`);

    const matchedDocs = new Set();
    const matches = [];

    for (const segment of segments) {
      const segmentVector = await this.embeddings.embedQuery(segment);
      const similarityScores = vectorDocs.map((docVector) => {
        const score = cosineSimilarity(segmentVector, docVector);
        return typeof score === "number" ? score : 0;
      });

      for (const [index, similarityScore] of similarityScores.entries()) {
        if (similarityScore <= THRESHOLD) {
          continue;
        }

        const relevantDoc = documents[index];
        if (matchedDocs.has(relevantDoc)) {
          continue;
        }

        matchedDocs.add(relevantDoc);

        const reviewResult = await reviewChain.invoke({
          segment,
          relevent_doc: relevantDoc,
        });

        const review = reviewResult.review;
        console.log(
          `Review: ${review} on the segment: ${segment} with the matched doc: ${relevantDoc}`
        );

        if (review === "positive") {
          matches.push([segment, relevantDoc]);
          console.log(`Segment: ${segment}`);
          console.log(`Matched with: ${relevantDoc}`);
        }

        console.log(`   Score: ${similarityScore.toFixed(2)}`);
      }
    }

    return matches;
  }
}

async function main() {
  const result = await new SemanticSearch().run();
  console.log(result, result.length);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
