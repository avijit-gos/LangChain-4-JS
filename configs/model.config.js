/** @format */

import { ChatGroq } from "@langchain/groq";
import { GROQ_API_KEY } from "../constants/index.js";

export const model = new ChatGroq({
  apiKey: GROQ_API_KEY,
  model: "llama-3.1-8b-instant",
  temperature: 0.7,
});
