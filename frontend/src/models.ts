export interface ModelEntry {
  id: string;
  label: string;
  family: string;
  group: string;
}

export const MODELS: ModelEntry[] = [
  { id: 'HuggingFaceTB/SmolLM2-135M', label: 'SmolLM2 135M · tiny, fast', family: 'llama', group: 'Small & fast' },
  { id: 'HuggingFaceTB/SmolLM2-360M', label: 'SmolLM2 360M', family: 'llama', group: 'Small & fast' },
  { id: 'HuggingFaceTB/SmolLM2-1.7B', label: 'SmolLM2 1.7B', family: 'llama', group: 'Small & fast' },
  { id: 'Qwen/Qwen2.5-0.5B-Instruct', label: 'Qwen2.5 0.5B · chat', family: 'qwen2', group: 'Qwen chat' },
  { id: 'Qwen/Qwen2.5-1.5B-Instruct', label: 'Qwen2.5 1.5B · chat', family: 'qwen2', group: 'Qwen chat' },
  { id: 'Qwen/Qwen2.5-3B-Instruct', label: 'Qwen2.5 3B · chat', family: 'qwen2', group: 'Qwen chat' },
  { id: 'Qwen/Qwen2.5-7B-Instruct', label: 'Qwen2.5 7B · needs 8GB+', family: 'qwen2', group: 'Qwen chat' },
  { id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', label: 'TinyLlama 1.1B · chat', family: 'llama', group: 'Popular' },
  { id: 'meta-llama/Llama-3.2-1B-Instruct', label: 'Llama 3.2 1B · gated', family: 'llama', group: 'Popular' },
  { id: 'meta-llama/Llama-3.2-3B-Instruct', label: 'Llama 3.2 3B · gated', family: 'llama', group: 'Popular' },
  { id: 'mistralai/Mistral-7B-Instruct-v0.3', label: 'Mistral 7B · needs 16GB+', family: 'mistral', group: 'Popular' },
  { id: 'google/gemma-2-2b-it', label: 'Gemma 2 2B · gated', family: 'gemma', group: 'Popular' },
  { id: 'microsoft/Phi-3-mini-4k-instruct', label: 'Phi-3 mini · MIT', family: 'phi', group: 'Popular' },
  { id: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', label: 'DeepSeek-R1 distill 1.5B', family: 'qwen2', group: 'Popular' },
];

export const GOALS: Array<[string, string, string]> = [
  ['balanced', 'Balanced', 'quality ~ size'],
  ['min-size', 'Min size', 'smallest disk'],
  ['max-quality', 'Max quality', 'lowest loss'],
  ['min-latency', 'Min latency', 'fastest tok/s'],
  ['cpu-efficient', 'CPU', 'no-GPU friendly'],
];

export const METHODS = ['auto', 'gptq', 'awq', 'hqq', 'gguf', 'bnb'];

export function shortModel(m: string): string {
  const p = m.split('/');
  return p.length > 1 ? p[1] : m;
}
