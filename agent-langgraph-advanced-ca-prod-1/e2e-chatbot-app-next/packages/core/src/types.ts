import { z } from 'zod';
import type { LanguageModelUsage, UIMessage } from 'ai';

const messageMetadataSchema = z.object({
  createdAt: z.string(),
});

type MessageMetadata = z.infer<typeof messageMetadataSchema>;

export const managedConversationDataSchema = z.object({
  id: z.string().startsWith('conv_'),
  created: z.boolean(),
});

export type ManagedConversationData = z.infer<
  typeof managedConversationDataSchema
>;

export type CustomUIDataTypes = {
  error: string;
  managedConversation: ManagedConversationData;
  usage: LanguageModelUsage;
  traceId: string | null;
  title: string;
};

export type ChatMessage = UIMessage<MessageMetadata, CustomUIDataTypes>;

export interface Attachment {
  name: string;
  url: string;
  contentType: string;
}

export type { VisibilityType } from '@chat-template/utils';

export interface Feedback {
  messageId: string;
  feedbackType: 'thumbs_up' | 'thumbs_down';
  assessmentId: string | null;
}

export type FeedbackMap = Record<string, Feedback>;
