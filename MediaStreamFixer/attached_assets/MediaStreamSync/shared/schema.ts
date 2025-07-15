import { pgTable, text, serial, integer, boolean, jsonb, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const videos = pgTable("videos", {
  id: serial("id").primaryKey(),
  filename: text("filename").notNull(),
  drillType: text("drill_type").notNull(),
  uploadedAt: timestamp("uploaded_at").defaultNow(),
  processed: boolean("processed").default(false),
});

export const analysisResults = pgTable("analysis_results", {
  id: serial("id").primaryKey(),
  videoId: integer("video_id").references(() => videos.id),
  analysisData: jsonb("analysis_data").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const chatMessages = pgTable("chat_messages", {
  id: serial("id").primaryKey(),
  videoId: integer("video_id").references(() => videos.id),
  message: text("message").notNull(),
  response: text("response").notNull(),
  timestamp: timestamp("timestamp").defaultNow(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertVideoSchema = createInsertSchema(videos).pick({
  filename: true,
  drillType: true,
});

export const insertAnalysisSchema = createInsertSchema(analysisResults).pick({
  videoId: true,
  analysisData: true,
});

export const insertChatMessageSchema = createInsertSchema(chatMessages).pick({
  videoId: true,
  message: true,
  response: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type InsertVideo = z.infer<typeof insertVideoSchema>;
export type Video = typeof videos.$inferSelect;
export type InsertAnalysisResult = z.infer<typeof insertAnalysisSchema>;
export type AnalysisResult = typeof analysisResults.$inferSelect;
export type InsertChatMessage = z.infer<typeof insertChatMessageSchema>;
export type ChatMessage = typeof chatMessages.$inferSelect;

export const DrillTypes = {
  PUSH_UPS: "push-ups",
  SQUATS: "squats",
  SIT_UPS: "sit-ups",
  WALL_SIT: "wall-sit",
  ELBOW_PLANK: "plank",
  SINGLE_LEG_BALANCE: "balance",
} as const;

export type DrillType = typeof DrillTypes[keyof typeof DrillTypes];
