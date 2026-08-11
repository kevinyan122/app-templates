import { expect, test } from '@playwright/test';
import {
  deleteManagedConversations,
  findManagedConversationIds,
} from '../../server/src/lib/managed-conversations';

test.describe('managed conversation cleanup', () => {
  const originalFetch = globalThis.fetch;
  const originalEnabled = process.env.DATABRICKS_CONVERSATIONS_ENABLED;
  const originalHost = process.env.DATABRICKS_HOST;

  test.afterEach(() => {
    globalThis.fetch = originalFetch;
    process.env.DATABRICKS_CONVERSATIONS_ENABLED = originalEnabled;
    process.env.DATABRICKS_HOST = originalHost;
  });

  test('finds unique valid conversation ids in stored message parts', () => {
    expect(
      findManagedConversationIds([
        {
          parts: [
            {
              type: 'data-managedConversation',
              data: { id: 'conv_first', created: true },
            },
          ],
        },
        {
          parts: [
            {
              type: 'data-managedConversation',
              data: { id: 'conv_first', created: false },
            },
            {
              type: 'data-managedConversation',
              data: { id: 'not-a-conversation', created: false },
            },
          ],
        },
      ]),
    ).toEqual(['conv_first']);
  });

  test('deletes each managed conversation with the forwarded user token', async () => {
    process.env.DATABRICKS_CONVERSATIONS_ENABLED = 'true';
    process.env.DATABRICKS_HOST = 'https://workspace.example.com/';
    const requests: Array<{ url: string; init?: RequestInit }> = [];

    globalThis.fetch = async (input, init) => {
      requests.push({ url: input.toString(), init });
      return new Response(null, { status: 204 });
    };

    await deleteManagedConversations({
      conversationIds: ['conv_first', 'conv_second', 'conv_first'],
      forwardedAccessToken: 'obo-token',
    });

    expect(requests).toHaveLength(2);
    expect(requests.map((request) => request.url).sort()).toEqual([
      'https://workspace.example.com/api/2.1/unity-catalog/conversations/conv_first',
      'https://workspace.example.com/api/2.1/unity-catalog/conversations/conv_second',
    ]);
    for (const request of requests) {
      expect(request.init?.method).toBe('DELETE');
      expect(request.init?.headers).toEqual({
        Authorization: 'Bearer obo-token',
      });
    }
  });

  test('does not call the API when managed conversations are disabled', async () => {
    process.env.DATABRICKS_CONVERSATIONS_ENABLED = 'false';
    let called = false;
    globalThis.fetch = async () => {
      called = true;
      return new Response(null, { status: 204 });
    };

    await deleteManagedConversations({
      conversationIds: ['conv_first'],
      forwardedAccessToken: undefined,
    });

    expect(called).toBe(false);
  });
});
