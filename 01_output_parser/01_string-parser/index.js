/** @format */

import { ChatPromptTemplate } from "@langchain/core/prompts";
import { model } from "../../configs/model.config.js";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { TextLoader } from "@langchain/classic/document_loaders/fs/text";
import path from "node:path";
import fs from "node:fs/promises";
import { fileURLToPath } from "url";
import { RecursiveCharacterTextSplitter } from "@langchain/classic/text_splitter";
import { Document } from "langchain";

/**
 * ✅ Problem 1: Basic text generation
 * Generate simple text responses from a given prompt using a language model.
 * Useful for tasks like content creation, explanations, or general responses.
 */
async function pattern_01() {
  try {
    const prompt = ChatPromptTemplate.fromTemplate(
      "Answer user's question in short format: {question}"
    );

    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    const res = await chain.invoke({
      question: "How can you help me?",
    });
    console.log(`# Response: ${res}`);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  }
}
// pattern_01();

/**
 * ✅ Problem 2: Streaming responses
 * Stream the model's output token-by-token instead of waiting for the full response.
 * Improves user experience by displaying responses in real time.
 */
async function pattern_02() {
  try {
    const prompt = ChatPromptTemplate.fromTemplate(
      "Provide answer to user's question: {question}"
    );
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    const chunks = await chain.stream({
      question: "Write a poem on river ganga.",
    });

    for await (const chunk of chunks) {
      process.stdout.write(chunk);
    }
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  }
}
// pattern_02();

/**
 * ✅ Problem 3: Summarization
 * Convert long text into a shorter version while preserving the main ideas.
 * Helpful for quickly understanding large documents or articles.
 */
async function createTextFile() {
  try {
    const text =
      `The River Ganga, also known as Ganges, is one of the most revered and sacred rivers in the world, flowing through the northern Indian subcontinent. Stretching approximately 2,525 kilometers from its origin in the Himalayas to its confluence with the Bay of Bengal, the Ganga is not only a vital source of water and livelihood for millions of people but also holds immense cultural, spiritual, and historical significance. Considered a sacred symbol of Hinduism, the river is a pilgrimage site for millions of devotees who come to bathe in its waters to attain spiritual cleansing and enlightenment. Its rich ecosystem supports an incredible array of flora and fauna, with various species of fish, birds, and other aquatic life calling the Ganga home. Despite facing numerous environmental challenges such as pollution and deforestation, efforts are being made to conserve and protect this precious natural resource, ensuring its survival for generations to come.`.repeat(
        10
      );
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const filepath = path.resolve(__dirname, "./sample.txt");
    const res = await fs.writeFile(filepath, text, "utf-8");
    console.log("=== Text file has been created ===");
  } catch (error) {
    console.log("Could not create text file", {
      message: error.message,
      stack: error.stack,
    });
  }
}
async function pattern_03() {
  try {
    let Summary = "";
    // text file created
    await createTextFile();
    const prompt = ChatPromptTemplate.fromMessages([
      ["system", "You are information summerization expert"],
      ["human", "Summarize following information {text} within one sentence"],
    ]);

    // load text file using textLoader
    const loader = new TextLoader("./sample.txt");
    const docs = await loader.load();

    // create a chain
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    // create document chunks
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 400,
      chunkOverlap: 50,
    });
    const chunks = await splitter.splitDocuments(docs);

    console.log(`=== Total number of chunks ${chunks.length} ===`);

    // lopping through the chunks
    for await (const [idx, chunk] of chunks.entries()) {
      console.log(`Processing chunk number: ${idx}...`);
      console.log(`=== Chunk Size: ${chunk.pageContent.length} ===\n`);
      const res = await chain.invoke({ text: chunk.pageContent });
      Summary += res + "\n";
    }
    console.log(Summary);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  }
}
// pattern_03();

/**
 * ✅ Problem 4: Translation
 * Translate text from one language to another using the language model.
 * Useful for multilingual applications and cross-language communication.
 */
async function pattern_04() {
  try {
    const prompt = ChatPromptTemplate.fromTemplate(
      "You are a translation assistance. Translate {message} from ${from_language} to {to_language}"
    );
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    const res = await chain.invoke({
      message: "Hello my friend",
      from_language: "English",
      to_language: "Spanish",
    });
    console.log(res);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  }
}
// pattern_04();

/**
 * ✅ Problem 5: Code explanation
 * Analyze a piece of code and generate a clear explanation of how it works.
 * Helps developers understand unfamiliar code or complex logic.
 */
async function pattern_05() {
  try {
    const prompt = ChatPromptTemplate.fromTemplate(
      "You are a code explanation expert. Explain {code} into simple terms"
    );

    // Load file
    const loader = new TextLoader("./utils.js");
    const docs = await loader.load();

    // create a chain
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    const res = await chain.invoke({
      code: docs,
    });
    console.log(res);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  }
}
// pattern_05();

/**
 * ✅ Problem 6: Question answering
 * Provide answers to user questions based on a given prompt or context.
 * Commonly used in chatbots, assistants, and knowledge retrieval systems.
 */
async function pattern_06() {
  try {
    const text = `The River Ganga, also known as Ganges, is one of the most revered and sacred rivers in the world, flowing through the northern Indian subcontinent. Stretching approximately 2,525 kilometers from its origin in the Himalayas to its confluence with the Bay of Bengal, the Ganga is not only a vital source of water and livelihood for millions of people but also holds immense cultural, spiritual, and historical significance. Considered a sacred symbol of Hinduism, the river is a pilgrimage site for millions of devotees who come to bathe in its waters to attain spiritual cleansing and enlightenment. Its rich ecosystem supports an incredible array of flora and fauna, with various species of fish, birds, and other aquatic life calling the Ganga home. Despite facing numerous environmental challenges such as pollution and deforestation, efforts are being made to conserve and protect this precious natural resource, ensuring its survival for generations to come.`;

    // create a Document
    const doc = new Document({
      pageContent: text,
      metadata: {
        source: "private.txt",
        timestamp: new Date().toISOString(),
      },
    });

    // create a prompt
    const prompt = ChatPromptTemplate.fromTemplate(
      `Anser user's question based on the provided context. If you cannot find answer within context then simply reply "I dont have the answer. context:  {context} and question: {question}`
    );

    // create a chain
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    const res = await chain.invoke({
      context: doc,
      question: "How long river ganga is?",
    });
    console.log(res);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  }
}
// pattern_06();

/**
 * ✅ Problem 7: Sentiment analysis (text response)
 * Determine the emotional tone of a piece of text (positive, negative, or neutral).
 * Useful for analyzing customer feedback, reviews, or social media content.
 */
async function pattern_07() {
  try {
    const prompt = ChatPromptTemplate.fromTemplate(
      `You are a sentiment assistant. Analysis {text} into POSITIVE or NEGATIVE sentiment`
    );
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    const res = await chain.invoke({
      text: "All team members were satisfied with the project outcome.",
    });
    console.log(res);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  }
}
// pattern_07();

/**
 * ✅ Problem 8: Text rewriting
 * Rewrite a given text while preserving the original meaning.
 * Often used for improving clarity, changing tone, or simplifying language.
 */
async function pattern_08() {
  try {
    const prompt = ChatPromptTemplate.fromTemplate(
      "You are a english expert. Rewrite user provided {text} with correct form also maintain actual meanig and emotion"
    );
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    const res = await chain.invoke({
      text: "I good boy",
    });
    console.log(res);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  }
}
// pattern_08();

/**
 * ✅ Problem 9: Title generation
 * Generate a short and relevant title based on a given text or article.
 * Helps create headlines for blogs, documents, or summaries.
 */
async function pattern_09() {
  try {
    // create a prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are title generation expert with having more than 10 years of experience. Generate a single title with provided user text. Also maintain actual emotion and generate title in formal way",
      ],
      ["human", "Generate title from {text}"],
    ]);

    // load text document
    const loader = new TextLoader("./sample.txt");
    const docs = await loader.load();

    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    const res = await chain.invoke({
      text: docs,
    });
    console.log(res);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  }
}
// pattern_09();

/**
 * ✅ Problem 10: Email generation
 * Automatically generate professional or casual email content from a prompt.
 * Useful for drafting replies, requests, or business communication.
 */
async function pattern_10() {
  try {
    // create a prompt
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are a email generation expert with having more than 10 years of experience. Generate an email for given topic including header title, email body in formal manner with correct emotion",
      ],
      ["human", "Generate an email from {topic}"],
    ]);
    const chain = prompt.pipe(model).pipe(new StringOutputParser());

    const res = await chain.invoke({
      topic: "Sick leave",
    });
    console.log(res);
  } catch (error) {
    console.log("Error:", {
      message: error.message,
      stack: error.stack,
    });
  }
}
// pattern_10();
