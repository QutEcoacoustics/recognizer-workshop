
import os, sys, argparse
import panel as pn
from recognizer_workshop.ui.panel.app import create_app_session


import logging, sys, os, asyncio
from pathlib import Path

def _setup_logging():
    logdir = Path.home() / ".recognizer-workshop" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / "app.log"

    # Root logger
    logging.basicConfig(
        level=logging.INFO,  # raise to DEBUG if needed
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Add a rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        logfile, maxBytes=5_000_000, backupCount=3
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    ))
    logging.getLogger().addHandler(fh)

    # Make Panel/Bokeh chatty
    logging.getLogger("bokeh").setLevel(logging.DEBUG)
    logging.getLogger("panel").setLevel(logging.DEBUG)

    # Optional: your package logger
    logging.getLogger("recognizer_workshop").setLevel(logging.DEBUG)

    # Catch otherwise-silent exceptions
    def _excepthook(exc_type, exc, tb):
        logging.getLogger("unhandled").exception("Unhandled exception", exc_info=(exc_type, exc, tb))
    sys.excepthook = _excepthook

    try:
        loop = asyncio.get_event_loop()
        def _handle_async_exc(loop, context):
            logging.getLogger("asyncio").error("Asyncio exception: %s", context)
        loop.set_exception_handler(_handle_async_exc)
    except Exception:
        pass

_setup_logging()
log = logging.getLogger("serve")




def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 0)))
    p.add_argument("--address", default=os.environ.get("ADDRESS", "127.0.0.1"))
    p.add_argument("--autoreload", action="store_true")
    p.add_argument("--singleton", action="store_true")
    args = p.parse_args(argv)

    def get_app():
        app, _vm = create_app_session()
        return app
    
    what_to_serve = get_app() if args.singleton else get_app

    pn.serve(
        {"/": what_to_serve},
        port=args.port,
        address=args.address,
        allow_websocket_origin=["*"],
        show=False,
        autoreload=args.autoreload,
    )

if __name__ == "__main__":
    main()
