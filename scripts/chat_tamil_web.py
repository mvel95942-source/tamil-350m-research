"""Temporary local streaming web chat for the 350M Tamil SFT preview.

Run: python scripts/chat_tamil_web.py
Open: http://127.0.0.1:8765
Override model path with TAMIL_MODEL_DIR. Uses CPU unless CUDA is available.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    import facts_ta
except ImportError:  # when run as a module from the repo root
    from scripts import facts_ta  # type: ignore

MODEL_DIR = Path(os.environ.get(
    "TAMIL_MODEL_DIR",
    str(Path.home() / "AppData/Local/Temp/opencode/kaggle-sft/sft_preview"),
))
HOST = "127.0.0.1"
PORT = int(os.environ.get("TAMIL_CHAT_PORT", "8765"))

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Merzal Tamil · Research Preview</title>
<style>
:root{color-scheme:dark;--bg:#101311;--panel:#171b18;--line:#2c342e;--text:#e6e9e4;--muted:#9ba69d;--lime:#c8f169;--user:#27351d}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 50% -15%,#24301e 0,transparent 45%),var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;min-height:100vh}
main{max-width:820px;margin:auto;padding:32px 20px 24px;display:flex;flex-direction:column;height:100vh}.top{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:18px}.brand{font-weight:750;letter-spacing:.02em}.brand span{color:var(--lime)}.badge{font:11px ui-monospace,monospace;color:var(--lime);border:1px solid #43552b;border-radius:99px;padding:5px 9px}.intro{padding:24px 0 14px}.intro h1{font-size:25px;line-height:1.2;margin:0 0 7px;letter-spacing:-.03em}.intro p{color:var(--muted);margin:0;font-size:13px}.notice{border:1px solid #55442a;background:#211d16;color:#d5c19d;padding:10px 12px;border-radius:9px;font-size:12px;margin:12px 0 18px}.chat{flex:1;overflow:auto;display:flex;flex-direction:column;gap:12px;padding:4px 2px 18px}.msg{max-width:88%;padding:12px 15px;border-radius:15px;white-space:pre-wrap;overflow-wrap:anywhere}.user{align-self:flex-end;background:var(--user);border:1px solid #43552b}.assistant{align-self:flex-start;background:var(--panel);border:1px solid var(--line)}.label{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:5px}.composer{border:1px solid var(--line);background:#171b18;border-radius:15px;padding:10px;display:flex;gap:10px;align-items:flex-end}.composer textarea{font:inherit;color:var(--text);background:transparent;border:0;outline:0;resize:none;flex:1;min-height:46px;max-height:140px;padding:10px}.composer textarea::placeholder{color:#778078}.composer button{border:0;border-radius:10px;background:var(--lime);color:#18200e;font-weight:750;padding:11px 17px;cursor:pointer}.composer button:disabled{opacity:.45;cursor:wait}.foot{display:flex;justify-content:space-between;color:#778078;font-size:11px;padding-top:10px}.link{color:var(--lime);cursor:pointer;background:none;border:0;font:inherit;padding:0}@media(max-width:600px){main{padding:20px 14px 14px}.msg{max-width:95%}}
</style></head><body><main>
<header class="top"><div class="brand">MERZAL <span>·</span> TAMIL LAB</div><div class="badge">350M · RESEARCH PREVIEW</div></header>
<section class="intro"><h1>Chat with the Tamil model</h1><p>A small from-scratch checkpoint, instruction-tuned for Tamil. Replies stream as they generate.</p></section>
<div class="notice">Research checkpoint for open-ended chat. Live/dated questions (PM, CM, army chief, population, weather, prices) are answered from a verified, dated source layer instead of the model. It can still be wrong on other topics.</div>
<section id="chat" class="chat" aria-live="polite"><div class="msg assistant"><div class="label">Merzal · research model</div>வணக்கம்! தமிழில் பேசிப் பாருங்கள். பிரதமர், முதல்வர், வானிலை போன்ற நேரடித் தகவல்களுக்கு சரிபார்க்கப்பட்ட ஆதாரத்துடன் பதில் தருவேன்.</div></section>
<form id="form" class="composer"><textarea id="input" rows="1" placeholder="உங்கள் செய்தியை எழுதுங்கள்…  (Enter to send)" aria-label="Message"></textarea><button id="send">Send</button></form>
<div class="foot"><span>Local session · conversation is not saved</span><button id="reset" class="link">Clear chat</button></div>
</main><script>
const form=document.querySelector('#form'),input=document.querySelector('#input'),chat=document.querySelector('#chat'),send=document.querySelector('#send');let history=[];
function bubble(role,text=''){const el=document.createElement('div');el.className='msg '+role;const label=document.createElement('div');label.className='label';label.textContent=role==='user'?'You':'Merzal · research model';const body=document.createElement('div');body.textContent=text;el.append(label,body);chat.append(el);chat.scrollTop=chat.scrollHeight;return body}
form.addEventListener('submit',async e=>{e.preventDefault();const text=input.value.trim();if(!text||send.disabled)return;input.value='';input.style.height='';bubble('user',text);const answer=bubble('assistant');send.disabled=true;input.disabled=true;
 try{const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({history,user:text})});if(!res.ok)throw new Error(await res.text());const reader=res.body.getReader(),decoder=new TextDecoder();let pending='',full='';while(true){const {value,done}=await reader.read();if(done)break;pending+=decoder.decode(value,{stream:true});const parts=pending.split('\n\n');pending=parts.pop();for(const part of parts){if(!part.startsWith('data: '))continue;const data=part.slice(6);if(data==='[DONE]')continue;const obj=JSON.parse(data);full+=obj.text;answer.textContent=full;chat.scrollTop=chat.scrollHeight}}history.push({role:'user',content:text},{role:'assistant',content:full});
 }catch(err){answer.textContent='Could not generate: '+err.message}finally{send.disabled=false;input.disabled=false;input.focus()}});
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit()}});input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,140)+'px'});
document.querySelector('#reset').addEventListener('click',()=>{history=[];chat.replaceChildren();bubble('assistant','உரையாடல் அழிக்கப்பட்டது. புதியதாகத் தொடங்கலாம்.')});
</script></body></html>'''


def render_prompt(history: list[dict[str, str]], user: str) -> str:
    messages = history[-12:] + [{"role": "user", "content": user}]
    parts = ["<s>"]
    for message in messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        marker = "<|user|>" if role == "user" else "<|assistant|>"
        parts.append(f"{marker}\n{message.get('content', '')}</s>")
    parts.append("<|assistant|>\n")
    return "".join(parts)


class Handler(BaseHTTPRequestHandler):
    tokenizer = None
    model = None
    _load_lock = threading.Lock()

    @classmethod
    def ensure_model(cls) -> None:
        if cls.model is not None:
            return
        with cls._load_lock:
            if cls.model is not None:
                return
            if not MODEL_DIR.is_dir():
                raise RuntimeError(f"model directory not found: {MODEL_DIR}")
            print(f"Loading checkpoint: {MODEL_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR, torch_dtype="auto", low_cpu_mem_usage=True, local_files_only=True
            ).eval()
            cls.tokenizer = tokenizer
            cls.model = model
            print(f"Loaded on {'CUDA' if torch.cuda.is_available() else 'CPU'}.")

    def log_message(self, fmt: str, *args: object) -> None:
        print("[chat] " + fmt % args)

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self.send_error(404)
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream_text(self, text: str) -> None:
        """Send a deterministic (grounded) answer over the same SSE channel."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        chunk = ("data: " + json.dumps({"text": text}, ensure_ascii=False) + "\n\n").encode("utf-8")
        self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
        self.wfile.flush()
        done = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(done):X}\r\n".encode("ascii") + done + b"\r\n0\r\n\r\n")
        self.wfile.flush()

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 100_000:
                self.send_error(413)
                return
            payload = json.loads(self.rfile.read(length))
            user = str(payload.get("user", "")).strip()
            history = payload.get("history", [])
            if not user or not isinstance(history, list):
                self.send_error(400, "Expected a user message and history list")
                return
            grounded = facts_ta.answer(user)
            if grounded:
                self.stream_text(grounded["text"])
                return
            try:
                self.ensure_model()
            except Exception as exc:
                self.stream_text(f"மாடலை ஏற்ற முடியவில்லை: {exc}")
                return
            encoded = self.tokenizer(
                render_prompt(history, user), return_tensors="pt", truncation=True,
                max_length=3500, add_special_tokens=False,
            )
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300
            )
            kwargs = dict(
                **encoded, streamer=streamer, max_new_tokens=192, do_sample=True,
                temperature=0.7, top_p=0.9, repetition_penalty=1.15,
                no_repeat_ngram_size=3, eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            worker = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
            worker.start()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for text in streamer:
                chunk = ("data: " + json.dumps({"text": text}, ensure_ascii=False) + "\n\n").encode("utf-8")
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
                self.wfile.flush()
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):X}\r\n".encode("ascii") + done + b"\r\n0\r\n\r\n")
            self.wfile.flush()
            worker.join()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            print(f"[chat error] {exc}")
            try:
                self.send_error(500, str(exc))
            except (BrokenPipeError, ConnectionResetError):
                pass


def main() -> None:
    facts_ta._load()
    facts_ta.refresh_in_background()
    print(f"Facts layer: {len(facts_ta._facts)} entries, snapshot {facts_ta._snapshot_date}")
    if MODEL_DIR.is_dir():
        print(f"Model loads on the first open-ended message: {MODEL_DIR}")
    else:
        print(f"WARNING: model directory not found: {MODEL_DIR}")
        print("Grounded facts/weather answers still work; open-ended chat will show an error.")
    print(f"Open http://{HOST}:{PORT} in your browser. Press Ctrl+C to stop.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
