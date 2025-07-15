import { users, videos, analysisResults, chatMessages, type User, type InsertUser, type Video, type InsertVideo, type AnalysisResult, type InsertAnalysisResult, type ChatMessage, type InsertChatMessage } from "@shared/schema";

export interface IStorage {
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  createVideo(video: InsertVideo): Promise<Video>;
  getVideo(id: number): Promise<Video | undefined>;
  updateVideoProcessed(id: number, processed: boolean): Promise<void>;
  
  createAnalysisResult(result: InsertAnalysisResult): Promise<AnalysisResult>;
  getAnalysisResultByVideoId(videoId: number): Promise<AnalysisResult | undefined>;
  
  createChatMessage(message: InsertChatMessage): Promise<ChatMessage>;
  getChatMessagesByVideoId(videoId: number): Promise<ChatMessage[]>;
}

export class MemStorage implements IStorage {
  private users: Map<number, User>;
  private videos: Map<number, Video>;
  private analysisResults: Map<number, AnalysisResult>;
  private chatMessages: Map<number, ChatMessage>;
  private currentUserId: number;
  private currentVideoId: number;
  private currentAnalysisId: number;
  private currentChatId: number;

  constructor() {
    this.users = new Map();
    this.videos = new Map();
    this.analysisResults = new Map();
    this.chatMessages = new Map();
    this.currentUserId = 1;
    this.currentVideoId = 1;
    this.currentAnalysisId = 1;
    this.currentChatId = 1;
  }

  async getUser(id: number): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = this.currentUserId++;
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async createVideo(insertVideo: InsertVideo): Promise<Video> {
    const id = this.currentVideoId++;
    const video: Video = { 
      ...insertVideo, 
      id,
      uploadedAt: new Date(),
      processed: false
    };
    this.videos.set(id, video);
    return video;
  }

  async getVideo(id: number): Promise<Video | undefined> {
    return this.videos.get(id);
  }

  async updateVideoProcessed(id: number, processed: boolean): Promise<void> {
    const video = this.videos.get(id);
    if (video) {
      video.processed = processed;
      this.videos.set(id, video);
    }
  }

  async createAnalysisResult(insertResult: InsertAnalysisResult): Promise<AnalysisResult> {
    const id = this.currentAnalysisId++;
    const result: AnalysisResult = { 
      ...insertResult, 
      id,
      createdAt: new Date()
    };
    this.analysisResults.set(id, result);
    return result;
  }

  async getAnalysisResultByVideoId(videoId: number): Promise<AnalysisResult | undefined> {
    return Array.from(this.analysisResults.values()).find(
      (result) => result.videoId === videoId,
    );
  }

  async createChatMessage(insertMessage: InsertChatMessage): Promise<ChatMessage> {
    const id = this.currentChatId++;
    const message: ChatMessage = { 
      ...insertMessage, 
      id,
      timestamp: new Date()
    };
    this.chatMessages.set(id, message);
    return message;
  }

  async getChatMessagesByVideoId(videoId: number): Promise<ChatMessage[]> {
    return Array.from(this.chatMessages.values()).filter(
      (message) => message.videoId === videoId,
    );
  }
}

export const storage = new MemStorage();
