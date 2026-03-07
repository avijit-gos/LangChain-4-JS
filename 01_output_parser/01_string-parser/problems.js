/** @format */

import { model } from "../../configs/model.config.js";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  Runnable,
  RunnableBranch,
  RunnableLambda,
  RunnableSequence,
} from "@langchain/core/runnables";
import { ChatGroq } from "@langchain/groq";
import { GROQ_API_KEY } from "../../constants/index.js";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/classic/text_splitter";

/**
 * ✅ Question 1: Basic String Output Parsing
 * Problem Description: You need to build a simple LangChain chain that takes a user's question and returns a plain string response from an LLM — not a complex object.
 */
async function pattern_01() {
  console.time("⏳ Execution Time:");
  try {
    const question = "How can you help me?";
    console.log(`Question: ${question}`);
    const prompt = ChatPromptTemplate.fromTemplate(
      "Answer user's question. user question: {question}, answer:"
    );
    const parser = new StringOutputParser();
    const chain = prompt.pipe(model).pipe(parser);

    const response = await chain.invoke({
      question,
    });
    console.log(`Answer: ${response} \n`);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time:");
  }
}
// pattern_01();

/**
 * ✅ Question 2: Comparing Raw LLM Output vs StringOutputParser Output
 * Problem Description: A junior developer on your team is confused about why calling .invoke() on a chain sometimes returns an AIMessage object instead of a plain string.
 */
async function pattern_02() {
  console.time("⏳ Execution Time:");
  try {
    const question = "What is LLM?";
    const prompt = ChatPromptTemplate.fromTemplate(
      "Answer user's question. user question: {question}, answer:"
    );

    const parser = new StringOutputParser();
    const chain = prompt.pipe(model).pipe(parser);

    const res = await chain.invoke({ question });
    console.log(`## Response from stringOutputParser: ${res} \n`);

    const res_1 = await prompt.pipe(model).invoke({ question });

    console.log(
      `## Response without stringOutputParser: ${JSON.stringify(
        res_1,
        null,
        2
      )}`
    );
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time:");
  }
}
// pattern_02();

/**
 * ✅ Question 3: Reusable Parser Utility Function
 * Problem Description: Your team wants a reusable helper that wraps any prompt string into a full chain (prompt → LLM → string output) so developers don't have to wire StringOutputParser every time.
 */
function buildStringChain(promptTemplate) {
  const parser = new StringOutputParser();
  const chain = promptTemplate.pipe(model).pipe(parser);
  return chain;
}
async function pattern_03() {
  console.time("⏳ Execution Time:");
  try {
    const prompt = ChatPromptTemplate.fromTemplate(
      "Answer user's question. user question: {question}, answer:"
    );
    /*
    const chain = buildStringChain(prompt);
    const response = await chain.invoke({
      question: "What is the full from of RAG in Gen-AI?",
    });
    console.log(response);
    */
    const response = await buildStringChain(prompt).invoke({
      question: "What is the full from of RAG in Gen-AI?",
    });
    console.log(response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time:");
  }
}
// pattern_03();

/**
 * ✅ Question 4: Streaming with StringOutputParser
 * Problem Description: You're building a chatbot UI that needs to stream tokens to the user one by one as the LLM generates them, rather than waiting for the full response.
 */
async function pattern_04() {
  console.time("⏳ Execution Time:");
  try {
    const prompt = ChatPromptTemplate.fromTemplate(
      "answer user's question. question: {question}, answer:"
    );

    const chunks = await prompt
      .pipe(model)
      .pipe(new StringOutputParser())
      .stream({ question: "Write a poem on moon" });

    for await (const chunk of chunks) {
      process.stdout.write(chunk);
    }
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time:");
  }
}
// pattern_04();

/**
 * ✅ Question 5: StringOutputParser Inside a Sequential Multi-Step Chain
 * Problem Description: You need to build a two-step content pipeline where the output of the first LLM call feeds directly into the second as plain text.
 */
async function pattern_05() {
  console.time("⏳ Execution Time:");
  try {
    // generate two prompts
    // first prompt is to generate Title of the product
    // second prompt is to generate sentiment
    const titlePrompt = ChatPromptTemplate.fromTemplate(
      "You are a title generation expert with having more than 10 years of experience. Generate maximum one single title for given topic: {topic}"
    );
    const sentimentPrompt = ChatPromptTemplate.fromTemplate(
      "You are sentiment checking expert. Analysis {title} into POSITIVE or NEGATIVE sentiment"
    );

    // ✅ Approach-1: pipe
    const titleChain = titlePrompt.pipe(model).pipe(new StringOutputParser());
    const sentimentChain = sentimentPrompt
      .pipe(model)
      .pipe(new StringOutputParser());

    const response = await titleChain
      .pipe((title) => ({ title }))
      .pipe(sentimentChain)
      .invoke({ topic: "Baby food product" });
    // console.log(response);

    // ✅ Approach-2: Using RunnableSequence
    /**
    const runnable_response = await RunnableSequence.from([
      titlePrompt,
      model,
      new StringOutputParser(),
      (title) => ({ title }),
      sentimentPrompt,
      model,
      new StringOutputParser(),
    ]).invoke({ topic: "Baby food product" });
    console.log(runnable_response);
     */

    const runnable_response = await RunnableSequence.from([
      titleChain,
      (title) => ({ title }),
      sentimentChain,
    ]).invoke({ topic: "Baby food product" });
    console.log(runnable_response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time:");
  }
}
// pattern_05();

/**
 * Question 6: Parsing and Post-Processing Output with .pipe()
 * Problem Description: You're building a keyword extraction service. The LLM returns keywords as a comma-separated string, and you need to transform that into a JavaScript array — all within a single LCEL chain.
 */
async function pattern_06() {
  console.time("⏳ Execution Time:");
  try {
    const prompt = ChatPromptTemplate.fromTemplate(
      `Extract exactly 5 keywords from the following text.
       Return ONLY a comma-separated list of keywords, nothing else.
       No numbering, no bullet points, no explanation.
       
       Example output: apple, banana, mango, grape, orange
       
       Text: {text}`
    );

    const splitKeywords = RunnableLambda.from((text) => {
      return text
        .split(",")
        .map((keyword) => keyword.trim())
        .filter((keyword) => keyword.length > 0);
    });

    const res = await prompt
      .pipe(model)
      .pipe(new StringOutputParser())
      .pipe(splitKeywords)
      .invoke({
        text: `Artificial intelligence is transforming industries by automating 
             tasks, improving decision-making, and enabling new capabilities 
             in healthcare, finance, and education.`,
      });
    console.log(res);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time:");
  }
}
// pattern_06();

/**
 * Question 7: Error Handling Around StringOutputParser in a Chain
 * Problem Description: In production, LLM calls can fail due to rate limits, timeouts, or invalid responses. Your chain should handle these gracefully without crashing the application.
 */
async function pattern_07() {
  console.time("⏳ Execution Time:");
  try {
    // create a prompt
    const prompt = ChatPromptTemplate.fromTemplate(
      "answer user's question. question: {question} and answer:"
    );

    // create an invalid LLM call
    const invalidModel = new ChatGroq({
      apiKey: GROQ_API_KEY,
      model: "xxx",
    });

    // create a chaiin with valid model
    const validChain = prompt.pipe(model).pipe(new StringOutputParser());

    // ===================================
    // ✅ Approcah-1: Model level fallback
    // ===================================
    /*
    const modelWithFallback = invalidModel.withFallbacks({
      fallbacks: [model],
    });
    const invalidModelchain = prompt
      .pipe(modelWithFallback)
      .pipe(new StringOutputParser());

    const response_1 = await invalidModelchain.invoke({
      question: "what is the capital of germany?",
    });
    console.log(response_1);
    */

    // ===================================
    // ✅ Approcah-2: Chain level fallback
    // ===================================

    const fallbackChain = RunnableLambda.from(() => {
      return "Sorry, your model name is not correct.";
    });
    const invalidModelchainCall = prompt
      .pipe(invalidModel)
      .pipe(new StringOutputParser());

    const chainWithFallbacks = invalidModelchainCall.withFallbacks({
      fallbacks: [fallbackChain],
    });

    const response = await chainWithFallbacks.invoke({
      question: "What is the capital of France?",
    });
    console.log(response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time:");
  }
}
// pattern_07();

/**
 * Question 8: Batch Processing with StringOutputParser and Concurrency Control
 */
async function pattern_08() {
  console.time("⏳ Execution Time:");
  try {
    let summarize = "";

    // create a prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are a strict summarization engine. You must follow instructions exactly.",
      ],
      [
        "user",
        `Summarize the following text in STRICTLY 10 to 20 words.
        Rules:
        - Return ONLY the summary.
        - No headings.
        - No bullet points.
        - No markdown.
        - No explanations.
        - No extra text.
        - Do NOT say "Here is the summary".
        - Do NOT exceed 20 words. Text: {text}`,
      ],
    ]);

    // load pdf document
    const loader = new PDFLoader("./sample_1.pdf", { splitPages: true });
    const docs = await loader.load();

    // initiate recursive text splitting
    const splitting = new RecursiveCharacterTextSplitter({
      chunkSize: 700,
      chunkOverlap: 100,
    });
    const chunks = await splitting.splitDocuments(docs);

    // create a prompt chain
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    let result = "";
    const BATCH_SIZE = 4;

    console.log(`=== Total number of chunks ${chunks.length} ===`);

    for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
      console.log(`✔︎ Process batch numbers ${Math.round(i / BATCH_SIZE) + 1}`);
      const batch = chunks.slice(i, i + BATCH_SIZE);
      const inputs = batch.map((chunk) => ({ text: chunk.pageContent }));
      const response = await chain.batch(inputs, { maxConcurrency: 4 });
      result += response.join("\n");
    }
    console.log(result);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time:");
  }
}
// pattern_08();

/**
 * Question 9: Dynamic Prompt Selection with StringOutputParser and RunnableBranch
 * Problem Description: You're building a multi-mode writing assistant. Depending on the mode field in the input ("formal", "casual", or "technical"), a different prompt template should be selected before passing to the LLM and then StringOutputParser.
 */
async function pattern_09() {
  console.time("⏳ Execution Time:");
  try {
    const content =
      "Our app helps users track their daily water intake and reminds them to stay hydrated.";

    const formalPrompt = ChatPromptTemplate.fromTemplate(
      `You are a formal business writer.
   Rewrite the following content in a formal, professional tone.
   Content: {content}`
    );

    const casualPrompt = ChatPromptTemplate.fromTemplate(
      `You are a friendly casual writer.
   Rewrite the following content in a fun, relaxed, conversational tone.
   Content: {content}`
    );

    const technicalPrompt = ChatPromptTemplate.fromTemplate(
      `You are a technical documentation expert.
   Rewrite the following content in a precise, technical tone with clear structure.
   Content: {content}`
    );

    const defaultPrompt = ChatPromptTemplate.fromTemplate(
      `You are a general purpose writer.
   Rewrite the following content in a clear and neutral tone.
   Content: {content}`
    );

    const prompt = RunnableBranch.from([
      [(input) => input.mode === "technical", technicalPrompt],

      [(input) => input.mode === "casual", casualPrompt],

      [(input) => input.model === "formal", formalPrompt],

      defaultPrompt,
    ]);

    const response = await prompt
      .pipe(model)
      .pipe(new StringOutputParser())
      .invoke({
        content,
        model: "technical",
      });
    console.log("Technical Response:", response);
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time:");
  }
}
// pattern_09();

/**
 * Question 10: Building a Stateful Conversation Chain with Memory and StringOutputParser
 * Problem Description: You need to build a multi-turn conversational chain that remembers previous exchanges. The assistant should respond in plain strings (not message objects), and the conversation history must be maintained across multiple .invoke() calls.
 */
async function pattern_10() {
  console.time("⏳ Execution Time:");
  try {
  } catch (error) {
    console.log("❌ Error:", {
      message: error.message,
      stack: error.stack,
    });
  } finally {
    console.timeEnd("⏳ Execution Time:");
  }
}
