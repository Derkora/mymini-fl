async function startTraining() {
    await fetch("/api/training/start", { method: "POST" });
    alert("Training started");
}

async function stopTraining() {
    await fetch("/api/training/stop", { method: "POST" });
    alert("Training stopped");
}

setInterval(async () => {
    const res = await fetch("/api/training/status");
    const data = await res.json();
    document.getElementById("status").innerHTML =
        `Running: ${data.running} | Round: ${data.current_round}`;
}, 1000);
