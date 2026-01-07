import { useEffect, useState } from "react";

export function useRemoteList({ url, keyName, errorMessage }) {
  const [items, setItems] = useState([]);
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let alive = true;

    async function load() {
      setLoading(true);
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(errorMessage);

        const data = await res.json();
        const list = Array.isArray(data?.[keyName]) ? data[keyName] : [];

        if (!alive) return;

        setItems(list);
        setSelected((prev) => (prev && list.includes(prev) ? prev : list[0] || ""));
      } catch (e) {
        if (!alive) return;
        throw e; // let caller decide how to show error
      } finally {
        if (alive) setLoading(false);
      }
    }

    load();

    return () => {
      alive = false;
    };
  }, [url, keyName, errorMessage]);

  return { items, selected, setSelected, loading, setItems };
}
