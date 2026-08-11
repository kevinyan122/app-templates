import { managedConversationDataSchema } from '@chat-template/core';
import { ChatSDKError } from '@chat-template/core/errors';

const CONVERSATIONS_PATH = '/api/2.1/unity-catalog/conversations';

type MessageWithParts = {
  parts: unknown;
};

function conversationsEnabled(): boolean {
  return ['1', 'true', 'yes'].includes(
    process.env.DATABRICKS_CONVERSATIONS_ENABLED?.trim().toLowerCase() ?? '',
  );
}

export function findManagedConversationIds(
  messages: MessageWithParts[],
): string[] {
  const conversationIds = new Set<string>();

  for (const message of messages) {
    if (!Array.isArray(message.parts)) continue;

    for (const part of message.parts) {
      if (
        typeof part !== 'object' ||
        part === null ||
        !('type' in part) ||
        part.type !== 'data-managedConversation' ||
        !('data' in part)
      ) {
        continue;
      }

      const parsed = managedConversationDataSchema.safeParse(part.data);
      if (parsed.success) conversationIds.add(parsed.data.id);
    }
  }

  return [...conversationIds];
}

export async function deleteManagedConversations({
  conversationIds,
  forwardedAccessToken,
}: {
  conversationIds: string[];
  forwardedAccessToken: string | undefined;
}): Promise<void> {
  if (!conversationsEnabled() || conversationIds.length === 0) return;

  const host = process.env.DATABRICKS_HOST?.trim().replace(/\/$/, '');
  if (!host || !forwardedAccessToken) {
    throw new ChatSDKError(
      'offline:chat',
      'Managed conversation deletion requires the workspace host and forwarded end-user token.',
    );
  }

  await Promise.all(
    [...new Set(conversationIds)].map(async (conversationId) => {
      const response = await fetch(
        `${host}${CONVERSATIONS_PATH}/${encodeURIComponent(conversationId)}`,
        {
          method: 'DELETE',
          headers: { Authorization: `Bearer ${forwardedAccessToken}` },
        },
      );

      if (!response.ok && response.status !== 404) {
        throw new ChatSDKError(
          'offline:chat',
          `Managed conversation deletion failed with status ${response.status}.`,
        );
      }
    }),
  );
}
