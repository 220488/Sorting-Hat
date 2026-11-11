"use client";

import { useRef } from "react";

export function ChartFigure({ src, alt }: { src: string; alt: string }) {
  const dialog = useRef<HTMLDialogElement>(null);

  return (
    <>
      <button
        type="button"
        className="figure"
        onClick={() => dialog.current?.showModal()}
        aria-label={`Enlarge: ${alt}`}
      >
        <img src={src} alt={alt} loading="lazy" />
      </button>
      <dialog
        ref={dialog}
        className="lightbox"
        onClick={() => dialog.current?.close()}
      >
        <figure style={{ margin: 0 }}>
          <img src={src} alt={alt} />
          <figcaption>{alt} · click or press Esc to close</figcaption>
        </figure>
      </dialog>
    </>
  );
}
