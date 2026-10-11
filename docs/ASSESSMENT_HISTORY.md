# Local assessment history (portfolio demo)

Campus Signals is a **synthetic demonstration**, not real-world screening. The Render service loads generated training examples; it does **not** persist submitted assessments or collect raw questionnaire answers.

After each successful prediction, the React app stores a **summary** in the current browser's `localStorage`: anonymous session ID, time, exploratory profile, behavioral cluster, anomaly decision value, derived social density and six normalized indicators. **It does not store the original 22 question answers.** Entries survive page refreshes and browser restarts on the same device, but are not shared or synchronized across devices or users. In incognito sessions or blocked browser storage, saving may fail and the UI reports it. Up to 250 most recent summaries are retained. Users can erase all summaries with **Clear history**.

The Overview dashboard updates its local trend, profile mix, last-session card, and recent history immediately after a successful assessment. Charts from synthetic training examples remain separated and explicitly labeled as demo benchmarks. A new assessment can be started immediately without refreshing the page.

The API's K-Means cluster label is descriptive. Isolation Forest returns a statistical outlier score, which is **not** a measure of isolation, a probability, or a clinical risk assessment. Exception logic does not turn an unusually high-engagement answer into a risk category merely because the answer is statistically uncommon. Self-reported difficulty and context guide *exploratory* supportive suggestions.

For shared, cross-device record-keeping in a legitimate campus program, this local history must NOT be silently replaced with a server database. Such a program requires explicit consent, a lawful basis, authentication, encryption, retention/deletion controls, privacy review, and independent validation.
