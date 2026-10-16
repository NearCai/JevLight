"""Keep long whole-step reads active only when a GPU chunk completes."""
import json
import threading
from jevlight.local_server import make_server


def make_step_server(backend, host, port, model_name):
    context=threading.local()
    class ProgressBackend:
        def __getattr__(self,name):
            return getattr(backend,name)
        def evaluate_batch(self,requests,batch_size):
            progress=backend.runtime._step_progress
            previous=getattr(progress,'notify',None)
            progress.notify=context.handler._progress
            try:
                return backend.evaluate_batch(requests,batch_size)
            finally:
                progress.notify=previous
    server=make_server(ProgressBackend(),host,port,model_name)
    parent=server.RequestHandlerClass
    class StepHandler(parent):
        protocol_version='HTTP/1.1'
        def do_POST(self):
            self._stream_started=False
            context.handler=self
            try:
                super().do_POST()
            finally:
                context.handler=None
        def _chunk(self,body):
            self.wfile.write(f'{len(body):X}\r\n'.encode()+body+b'\r\n')
            self.wfile.flush()
        def _progress(self):
            if not self._stream_started:
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Transfer-Encoding','chunked')
                self.send_header('Connection','close')
                self.end_headers()
                self._stream_started=True
            # Whitespace is valid before the final JSON. No timer heartbeat:
            # a hung GPU chunk still triggers the existing client read timeout.
            self._chunk(b'\n')
        def _send(self,status,payload):
            if getattr(self,'_stream_started',False):
                self._chunk(json.dumps(payload,ensure_ascii=False).encode())
                self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
                self.close_connection=True
            else:
                super()._send(status,payload)
    server.RequestHandlerClass=StepHandler
    return server
