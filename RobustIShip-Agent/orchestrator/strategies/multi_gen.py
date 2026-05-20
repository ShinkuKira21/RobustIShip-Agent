"""Multi-gen strategy — 3 Qwen + 1 Gemma variants, Gemma assembles best. (EXPERIMENTAL)"""

import concurrent.futures
import json

from tools.files import write_file
from tools.validate import validate_written_file
from tools.normalize import _fix_absolute_path
from qwen.actions import request_qwen_action


def parse_gemma_raw(raw: str) -> tuple[str, str] | None:
    if not raw:
        return None
    import re
    lines = raw.strip().split('\n')
    if not lines:
        return None
    path = None
    content_start = 0
    if lines[0].startswith("PATH:"):
        path = lines[0][5:].strip()
        content_start = 1
        if content_start < len(lines) and lines[content_start].strip() == "":
            content_start += 1
    if not path:
        first_line = lines[0].strip()
        if first_line.endswith(".py") and "/" not in first_line and " " not in first_line:
            path = first_line
            content_start = 1
    if not path:
        return None
    content = '\n'.join(lines[content_start:]).strip()
    if not content:
        return None
    content = re.sub(r'^```\w*\n?', '', content)
    content = re.sub(r'\n?```$', '', content)
    return path, content


def handle(gemma_model, args, root, state, flags, step_index,
           step_prompt, expected_tool, injected_context, observations,
           path, content) -> dict:
    print("   🧬 Multi-Gen: 3 Qwen variants + 1 Gemma variant — assembling best...")

    def qwen_variant(seed):
        varied = f"{step_prompt}\n\n[APPROACH: {seed}]"
        return request_qwen_action(
            args, root, expected_tool, varied,
            injected_context=injected_context, observations=observations,
            step_index=step_index, purpose=f"multi_qwen_{seed[:10]}",
        )

    def gemma_variant():
        reflection_context = state.assemble_reflection_context(step_index, "", current_tool=expected_tool)
        prompt = (
            f"Write the COMPLETE file for: {step_prompt}\n\n"
            f"CONTEXT:\n{json.dumps(reflection_context, indent=2, default=str)[:1500]}\n\n"
            f"Output: PATH: <path>\n\n<content>"
        )
        messages = [
            {"role": "system", "content": "Write complete, correct code."},
            {"role": "user", "content": prompt},
        ]
        try:
            return gemma_model.generate(messages, max_tokens=2048, temperature=0.0)
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(qwen_variant, "robustness"),
            executor.submit(qwen_variant, "cleanliness"),
            executor.submit(qwen_variant, "testability"),
            executor.submit(gemma_variant),
        ]
        results = []
        for f in futures:
            try:
                results.append(f.result(timeout=flags.multi_gen_timeout))
            except Exception:
                results.append(None)

    variants = []
    for r in results[:3]:  # Qwen results
        if r and isinstance(r, dict) and "final" not in r:
            variants.append(r)
    if results[3]:  # Gemma result
        parsed = parse_gemma_raw(results[3])
        if parsed:
            gp, gc = parsed
            variants.append({"tool": "write_file", "args": {"path": gp, "content": gc}})

    if len(variants) < 2:
        print("   ⚠️  Not enough variants for assembly — falling back to first available")
        if variants:
            v = variants[0]
            write_file(root, v["args"]["path"], v["args"]["content"])
            state.store_file_content(v["args"]["path"], v["args"]["content"])
            state.mark_done_by_index(step_index)
            return {"ok": True, "summary": f"Fallback single variant: {v['args']['path']}"}
        return {"ok": False, "summary": "Multi-gen: no valid variants produced"}

    # Gemma assembles the best version
    variants_text = ""
    for i, v in enumerate(variants):
        vpath = v.get("args", {}).get("path", "unknown")
        vcontent = v.get("args", {}).get("content", "")[:1200]
        variants_text += f"\n=== VARIANT {i+1} ===\nFILE: {vpath}\n{vcontent}\n"

    assembly_prompt = (
        f"TASK: {step_prompt}\n\n"
        f"Review {len(variants)} code variants:\n{variants_text}\n\n"
        f"Select the BEST parts from each. Assemble a FINAL version (target 10/10).\n"
        f"Output:\nPATH: <path>\n\n<complete content>"
    )
    assembly_messages = [
        {"role": "system", "content": "You are a senior code reviewer. Produce a flawless final version."},
        {"role": "user", "content": assembly_prompt},
    ]

    try:
        assembly_raw = gemma_model.generate(assembly_messages, max_tokens=2048, temperature=0.0)
        parsed = parse_gemma_raw(assembly_raw)
        if parsed:
            final_path, final_content = parsed
            write_file(root, final_path, final_content)
            state.store_file_content(final_path, final_content)
            validation = validate_written_file(root, _fix_absolute_path(final_path, root), pre_flight=flags.pre_flight)
            if validation.get("ok"):
                state.mark_done_by_index(step_index)
                return {"ok": True, "summary": f"Assembled from {len(variants)} variants: {final_path}"}
            return {"ok": False, "summary": f"Assembly failed validation: {validation.get('stderr', '')}"}
    except Exception as e:
        print(f"   ❌ Assembly error: {e}")

    # Fallback: use first variant
    v = variants[0]
    write_file(root, v["args"]["path"], v["args"]["content"])
    state.store_file_content(v["args"]["path"], v["args"]["content"])
    state.mark_done_by_index(step_index)
    return {"ok": True, "summary": f"Fallback: {v['args']['path']}"}