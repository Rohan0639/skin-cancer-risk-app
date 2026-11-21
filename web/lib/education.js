/**
 * Patient education content for the result screen.
 *
 * Kept in one file, separate from the model and the pipeline, so a clinician
 * can review and revise the wording without reading any inference code. Nothing
 * here touches the model, the prediction or the risk tier: it reads the report
 * the pipeline already produced and selects text.
 *
 * Safety rules this module enforces, not merely documents:
 *   - A screening result is never written as a diagnosis.
 *   - No stage is ever produced. There is no staging logic here to produce one.
 *   - No claim is made about spread, in either direction.
 *   - No treatment is recommended for the reader; treatments are described.
 *   - No survival figure appears anywhere.
 *   - A benign result never says "clear" or "cancer-free".
 *   - When the model abstains, no condition-specific content is shown at all.
 *
 * Content follows patient-facing guidance from the American Academy of
 * Dermatology, the National Cancer Institute, the American Cancer Society and
 * the NHS. It is general education, not advice about this reader's lesion.
 */

/** Per-class education. `klass` drives wording, never a stage or spread claim. */
const CONDITIONS = {
  NEV: {
    full: 'Melanocytic naevus (a common mole)',
    klass: 'benign',
    klassLabel: 'Generally benign',
    what: `A mole is a harmless cluster of pigment-producing cells. Most adults
      have between 10 and 40 of them. They usually appear in childhood and early
      adulthood, and can slowly change over many years, which on its own is
      normal.`,
    features: [
      'Usually round or oval, with a smooth and even border',
      'Usually one shade of brown, tan or pink throughout',
      'Usually smaller than about 6 mm, though larger moles can still be harmless',
      'Usually looks similar to the other moles on the same person',
    ],
    whyExam: `A small number of melanomas begin in or near an existing mole, and
      early melanoma can look very like an ordinary mole in a photograph. A
      clinician examining the skin directly — usually with a dermatoscope — can
      see structure that no phone camera captures.`,
    urgent: `Ask for an appointment if a mole changes shape, colour or size,
      becomes itchy or painful, bleeds, crusts, or starts to look different from
      your other moles.`,
  },
  SEK: {
    full: 'Seborrhoeic keratosis',
    klass: 'benign',
    klassLabel: 'Generally benign',
    what: `A seborrhoeic keratosis is a harmless growth that becomes more common
      with age. It is not caused by sun damage and it does not turn into cancer.
      People often have several. They are frequently mistaken for something more
      serious because they can be dark and irregular.`,
    features: [
      'Often looks waxy, scaly or "stuck on" the skin surface',
      'Colour ranges from pale tan to very dark brown or black',
      'Surface may be rough, warty or cracked',
      'Common on the chest, back, face and scalp',
    ],
    whyExam: `A pigmented seborrhoeic keratosis can closely resemble a melanoma,
      and the two are told apart by features visible under a dermatoscope rather
      than by eye. That distinction matters, so a growth read this way is still
      worth showing to a clinician if you are unsure.`,
    urgent: `Ask for an appointment if the growth bleeds, becomes painful, grows
      quickly, or if many appear suddenly over a short period.`,
  },
  ACK: {
    full: 'Actinic keratosis (solar keratosis)',
    klass: 'precancerous',
    klassLabel: 'Pre-cancerous — not a cancer, but not harmless',
    what: `An actinic keratosis is an area of sun-damaged skin. It is not a
      cancer. It is considered pre-cancerous because a small proportion, if left
      untreated, can develop into squamous cell carcinoma over time. Having them
      also indicates cumulative sun damage, which raises the chance of other skin
      cancers appearing elsewhere.`,
    features: [
      'Rough, dry or scaly patch, often easier to feel than to see',
      'Pink, red or skin-coloured, sometimes with a sandpapery surface',
      'Found on sun-exposed skin: face, scalp, ears, forearms, backs of hands',
      'Often several are present at once',
    ],
    whyExam: `A clinician can distinguish an actinic keratosis from an early
      squamous cell carcinoma, which can look similar. Treatment is usually
      straightforward, and treating these lesions is partly about preventing a
      later cancer.`,
    urgent: `Ask for an appointment sooner if a patch thickens, becomes tender or
      painful, grows into a lump, bleeds, or does not heal.`,
  },
  BCC: {
    full: 'Basal cell carcinoma',
    klass: 'malignant',
    klassLabel: 'A form of skin cancer',
    what: `Basal cell carcinoma is the most common form of skin cancer. It grows
      slowly and very rarely spreads to other parts of the body. Left untreated
      it can grow into nearby tissue and cause local damage, which is why it is
      treated rather than monitored. Treated in good time, the outlook is
      generally very good.`,
    features: [
      'Pearly or waxy bump, sometimes with small visible blood vessels',
      'Flat, scaly patch that may be pink or pale',
      'A sore that bleeds, scabs, heals and then returns in the same place',
      'Most common on the face, ears, neck and other sun-exposed skin',
    ],
    whyExam: `Confirming a basal cell carcinoma normally requires a clinician to
      examine the lesion and usually to take a biopsy. The subtype and the
      location both affect which treatment is appropriate, and neither can be
      determined from a photograph.`,
    urgent: `Arrange an appointment promptly. This is not usually an emergency,
      but it should not be left.`,
  },
  SCC: {
    full: 'Squamous cell carcinoma',
    klass: 'malignant',
    klassLabel: 'A form of skin cancer',
    what: `Squamous cell carcinoma is the second most common skin cancer. Most
      cases are cured when found and treated early. Unlike basal cell carcinoma
      it has a greater capacity to spread to lymph nodes or beyond, particularly
      when the lesion is large, deep, on the lip or ear, or when the person's
      immune system is suppressed. That is why timely assessment matters.`,
    features: [
      'Firm red lump, or a flat sore with a scaly crust',
      'A sore or rough patch that does not heal over several weeks',
      'May be tender, may bleed or crust repeatedly',
      'Common on sun-exposed skin, and on the lip and ear',
    ],
    whyExam: `Diagnosis is confirmed by biopsy. A clinician also assesses
      features that affect risk — size, depth, location, how the cells look under
      a microscope — none of which are visible in a photograph.`,
    urgent: `Arrange an appointment promptly. Say on the phone that you have a
      sore or lump that is not healing, so it can be prioritised appropriately.`,
  },
  MEL: {
    full: 'Melanoma',
    klass: 'malignant',
    klassLabel: 'A form of skin cancer',
    what: `Melanoma is less common than other skin cancers but more serious,
      because it can spread to other parts of the body if not treated early. When
      melanoma is found early, treatment is usually very effective. This is the
      result that most warrants prompt professional assessment.`,
    features: [
      'A new mole, or an existing mole that is changing',
      'Asymmetry: the two halves do not match',
      'An irregular, notched or blurred border',
      'More than one colour: browns, blacks, reds, whites or blue-greys',
      'Often, though not always, larger than about 6 mm',
      'Any mole that itches, bleeds, crusts or looks unlike your others',
    ],
    whyExam: `Melanoma is confirmed by removing the lesion and examining it under
      a microscope. The thickness measured at that point is the single most
      important factor in what happens next, and it cannot be judged from a
      photograph.`,
    urgent: `Arrange an appointment as soon as you can. When you book, say you
      are concerned about a changing mole — many services prioritise this.`,
  },
};

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const li = (items) => items.map((t) => `<li>${esc(t)}</li>`).join('');

/** One accordion card. Native <details>, so it is keyboard and screen-reader friendly. */
function card(n, title, body, open = false) {
  return `<details class="edu-card"${open ? ' open' : ''}>
    <summary><span class="edu-n">${n}</span><span>${esc(title)}</span></summary>
    <div class="edu-body">${body}</div>
  </details>`;
}

function sectionTitle(t) {
  return `<h4 class="edu-sec">${esc(t)}</h4>`;
}

// --- Card 2: staging -------------------------------------------------------
// There is deliberately no input from the prediction here beyond which general
// explanation to show. Nothing in this function can produce a stage.
function stagingCard(cond) {
  const melanomaStages = `
    <p>For melanoma, stages are described in general terms as follows. These are
      explanations of the terminology, not an assessment of your lesion.</p>
    <ul class="edu-stage">
      <li><b>Stage 0</b> — melanoma in situ: abnormal cells are confined to the
        outermost layer of skin.</li>
      <li><b>Stage I</b> — early invasive melanoma, thin, with no sign of spread.</li>
      <li><b>Stage II</b> — thicker melanoma, with no identified lymph node or
        distant spread.</li>
      <li><b>Stage III</b> — spread to nearby lymph nodes or nearby tissue.</li>
      <li><b>Stage IV</b> — spread to distant organs.</li>
    </ul>`;
  const keratinocyte = `
    <p>For basal cell and squamous cell carcinoma, formal staging is used less
      often than for melanoma. Instead a clinician assesses whether the lesion is
      low or high risk, using factors such as its size, its depth, where it is on
      the body, whether it has been treated before, how the cells appear under a
      microscope, and whether the immune system is suppressed.</p>`;

  const body = `
    <p class="edu-flag"><b>Your stage cannot be determined from this AI screening
      result.</b> Staging follows a clinical examination and a biopsy, and
      sometimes further investigations. It is not something that can be read from
      a photograph by any system, including this one.</p>
    <p>Staging describes how deep a cancer has grown and whether it has spread. It
      is established by a doctor using the tissue removed at biopsy, measurements
      taken under a microscope, and examination findings.</p>
    ${cond && cond.klass === 'malignant'
      ? (cond.full === 'Melanoma' ? melanomaStages : keratinocyte)
      : `<p>Staging applies to confirmed cancers. ${cond && cond.klass === 'precancerous'
          ? 'An actinic keratosis is a pre-cancerous change and is not staged.'
          : 'Benign lesions are not staged.'}</p>` + melanomaStages}
    <p class="edu-note">Only your treating clinician can tell you a stage, and
      only after the relevant tests.</p>`;
  return card(2, 'What is its stage?', body);
}

// --- Card 3: spread --------------------------------------------------------
function spreadCard() {
  return card(3, 'Has it spread?', `
    <p class="edu-flag"><b>This screening result cannot tell you whether anything
      has spread</b> — in either direction. A photograph shows the surface of the
      skin and nothing beneath it.</p>
    <p>If a cancer is confirmed, these are the terms you may hear:</p>
    <ul>
      <li><b>Local</b> — the cancer is confined to where it started, growing into
        the surrounding skin or tissue.</li>
      <li><b>Regional / lymph node involvement</b> — cells have reached nearby
        lymph nodes, the small glands that drain fluid from that area.</li>
      <li><b>Distant metastasis</b> — cells have travelled to organs further away,
        such as the lungs, liver, bones or brain.</li>
    </ul>
    <p>Depending on the confirmed diagnosis, a dermatologist or oncologist may
      recommend further examination, assessment of nearby lymph nodes, imaging, or
      other tests. Many people need none of these: for most basal cell carcinomas,
      for example, spread is very rare.</p>`);
}

// --- Card 4: treatment -----------------------------------------------------
function treatmentCard() {
  return card(4, 'What treatment do I need?', `
    <p class="edu-flag">The treatments below are described so the words are
      familiar when you hear them. <b>This is not a recommendation, and no
      treatment should be chosen from an AI screening result.</b></p>
    <ul class="edu-tx">
      <li><b>Surgical excision</b> — the lesion is cut out along with a margin of
        normal-looking skin around it. The most common treatment for skin cancer.</li>
      <li><b>Mohs micrographic surgery</b> — tissue is removed in thin layers and
        examined immediately, repeating until no abnormal cells remain. Used where
        sparing healthy tissue matters, such as on the face.</li>
      <li><b>Cryotherapy</b> — the lesion is frozen with liquid nitrogen. Often
        used for actinic keratoses and some superficial lesions.</li>
      <li><b>Curettage and electrodessication</b> — the lesion is scraped away and
        the base treated with heat.</li>
      <li><b>Topical treatments</b> — creams applied over several weeks, used for
        some superficial lesions and for actinic keratoses.</li>
      <li><b>Photodynamic therapy</b> — a light-sensitive agent is applied and
        then activated with a specific light source.</li>
      <li><b>Radiation therapy</b> — targeted radiation, used where surgery is not
        suitable or as an additional treatment.</li>
      <li><b>Immunotherapy</b> — medicines that help the immune system recognise
        and attack cancer cells. Mainly used for advanced melanoma and some
        advanced skin cancers.</li>
      <li><b>Targeted therapy</b> — medicines directed at specific genetic changes
        found in some melanomas.</li>
    </ul>
    <p>Which of these applies depends on the confirmed type of lesion, its stage,
      its size and position, how deeply it has grown, and your general health and
      preferences. Two people with the same diagnosis can appropriately receive
      different treatment.</p>`);
}

// --- Card 5: side effects --------------------------------------------------
function sideEffectsCard() {
  const rows = [
    ['Surgical excision', 'Pain, bleeding, infection, scarring'],
    ['Mohs surgery', 'Temporary swelling, discomfort, scarring'],
    ['Cryotherapy', 'Blistering, redness, lasting pigment changes'],
    ['Curettage', 'Scarring, pigment change at the site'],
    ['Topical treatments', 'Redness, soreness, crusting while in use'],
    ['Photodynamic therapy', 'Stinging during treatment, redness, light sensitivity'],
    ['Radiation therapy', 'Skin irritation, fatigue, longer-term local tissue change'],
    ['Immunotherapy', 'Inflammation and immune-related effects, which can affect several organs'],
    ['Targeted therapy', 'Skin reactions, fatigue, and effects specific to the medicine'],
  ].map(([t, s]) => `<tr><td>${esc(t)}</td><td>${esc(s)}</td></tr>`).join('');

  return card(5, 'What are the possible side effects?', `
    <div class="edu-scroll"><table class="edu-table">
      <thead><tr><th>Treatment</th><th>Possible side effects</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p>Side effects differ a great deal between people and between treatments, and
      most are temporary. This list is not complete, and it is not a prediction of
      what you would experience.</p>
    <p class="edu-flag"><b>If you are having treatment and develop symptoms that
      concern you — a wound that becomes hot, swollen or increasingly painful, a
      fever, or any sudden change in how you feel — contact your treating team.</b></p>`);
}

// --- Card 6: follow-up -----------------------------------------------------
function followUpCard() {
  return card(6, 'How often will I need follow-up?', `
    <p class="edu-flag"><b>Your dermatologist will set your personal follow-up
      schedule.</b> There is no single interval that suits everyone, and this tool
      cannot work one out.</p>
    <p>In general terms:</p>
    <ul>
      <li>After treatment for a <b>low-risk skin cancer</b>, periodic skin checks
        are usual, at intervals your dermatologist decides.</li>
      <li>After a <b>higher-risk basal or squamous cell carcinoma</b>, follow-up
        is usually closer, particularly in the first years.</li>
      <li>After <b>melanoma</b>, how often and for how long depends on the stage
        and on the risk of recurrence, and may include examination of lymph nodes.</li>
      <li>For <b>actinic keratoses</b>, review tends to focus on new lesions
        appearing, since sun-damaged skin often produces more.</li>
    </ul>
    <p>Between appointments, checking your own skin regularly matters as much as
      the appointments themselves. Most skin cancers are first noticed by the
      person who has them, or by someone close to them.</p>`);
}

// --- Card 7: future risk ---------------------------------------------------
function futureRiskCard() {
  return card(7, 'What is my risk of another skin cancer?', `
    <p>Risk of developing a skin cancer in future is higher for people with:</p>
    <ul>
      <li>A previous skin cancer — the single strongest factor</li>
      <li>A family history of melanoma or other skin cancer</li>
      <li>Substantial lifetime sun exposure, or a history of sunburn</li>
      <li>Many moles, or moles that are unusual in appearance</li>
      <li>Fair skin that burns easily, light eyes, red or fair hair</li>
      <li>A suppressed immune system, including after an organ transplant</li>
      <li>Previous radiation treatment to the skin</li>
    </ul>
    <p>What reduces risk:</p>
    <ul>
      <li>Broad-spectrum sunscreen, SPF 30 or above, reapplied every two hours
        outdoors and after swimming</li>
      <li>Clothing cover, a wide-brimmed hat and sunglasses</li>
      <li>Shade in the middle of the day, when sunlight is most intense</li>
      <li>No sunbeds or tanning lamps, at any age</li>
      <li>Checking your own skin about once a month, including places you do not
        easily see — use a mirror, or ask someone</li>
      <li>Keeping the dermatology appointments you are offered</li>
    </ul>
    <p class="edu-note">If you have had a skin cancer before, continued monitoring
      matters more, not less. A previous skin cancer raises the chance of another
      one appearing, often somewhere different.</p>`);
}

/**
 * Build the education section for a finished report.
 *
 * @param {object} report   the pipeline's report object
 * @param {boolean} uncertain  true when the model declined to name a diagnosis
 * @returns {string} HTML
 */
export function renderEducation(report, uncertain) {
  const code = report && report.diagnosis;
  const cond = (!uncertain && code) ? CONDITIONS[code] : null;

  // --- 1. Your screening result -------------------------------------------
  const resultBlock = cond
    ? `<p>Your image screening model identified features most consistent with
        <b>${esc(cond.full)}</b>. <b>This is an AI-generated screening result, not
        a confirmed diagnosis.</b> A dermatologist may recommend an examination
        with a dermatoscope, or a biopsy, to establish what this actually is.</p>
       <p class="edu-klass edu-${cond.klass}">${esc(cond.klassLabel)}</p>`
    : `<p class="edu-flag"><b>This image could not be assessed reliably.</b> The
        model did not reach enough confidence to suggest a lesion type, so no
        condition-specific information is shown below — offering any would mean
        guessing.</p>
       <p>The general information in the cards below still applies to skin lesions
        as a whole. If you are at all concerned about this lesion, arrange to have
        it looked at rather than waiting for a clearer photograph.</p>`;

  // --- 2. Understanding your condition (Card 1) ---------------------------
  const card1Body = cond
    ? `<p>${esc(cond.what).replace(/\s+/g, ' ')}</p>
       <p class="edu-klass edu-${cond.klass}">${esc(cond.klassLabel)}</p>
       <p><b>What it commonly looks like</b></p>
       <ul>${li(cond.features)}</ul>
       <p><b>Why a professional examination matters</b></p>
       <p>${esc(cond.whyExam).replace(/\s+/g, ' ')}</p>
       <p class="edu-flag">${esc(cond.urgent).replace(/\s+/g, ' ')}</p>`
    : `<p>No lesion type is shown, because the screening result was not confident
        enough to suggest one. The six types this tool can recognise are moles and
        seborrhoeic keratoses (generally benign), actinic keratoses
        (pre-cancerous), and basal cell carcinoma, squamous cell carcinoma and
        melanoma (skin cancers).</p>
       <p>Only an examination can establish which, if any, applies here.</p>`;

  return `
  <section class="edu" aria-label="Understanding your diagnosis and next steps">
    <h3 class="edu-title">Understanding Your Diagnosis &amp; Next Steps</h3>
    <p class="edu-intro">General information to help you prepare for a
      conversation with a clinician. It is education, not advice about your
      lesion, and it does not replace a professional opinion.</p>

    ${sectionTitle('1. Your screening result')}
    <div class="edu-result">${resultBlock}</div>

    ${sectionTitle('2. Understanding your condition')}
    ${card(1, 'What exact type of lesion is this?', card1Body, true)}

    ${sectionTitle('3. Diagnosis and staging')}
    ${stagingCard(cond)}
    ${spreadCard()}

    ${sectionTitle('4. Treatment information')}
    ${treatmentCard()}

    ${sectionTitle('5. Possible side effects')}
    ${sideEffectsCard()}

    ${sectionTitle('6. Follow-up and prevention')}
    ${followUpCard()}
    ${futureRiskCard()}

    <div class="edu-cta">
      <b>Discuss this with your dermatologist.</b>
      <p>Take this screening result and the photograph with you. Mention how long
        the lesion has been there and anything that has changed about it — that
        history is often more useful to a clinician than the image itself.</p>
    </div>

    <p class="edu-src">Educational content follows patient guidance from the
      American Academy of Dermatology, the National Cancer Institute, the American
      Cancer Society and the NHS. It is general information, has not been reviewed
      for your individual case, and may become out of date.</p>
  </section>`;
}
