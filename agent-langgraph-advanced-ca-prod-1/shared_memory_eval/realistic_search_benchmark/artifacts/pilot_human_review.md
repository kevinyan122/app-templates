# Realistic memory-search pilot: human review

Source: `pilot_enriched.json`  
Label sheet: `pilot_human_labels.csv`  
Cases: 25  
Corpus memories: 60  
Benchmark as of: `2026-06-05`  
Source generated at: `2026-08-04T21:34:16.344047+00:00`

## How to review

Review all 25 cases without opening `pilot_enriched_reviews.json`. For each case, decide:

1. Does the query sound natural and have one clear interpretation?
2. Does the gold memory alone support every material part of `answer_should`?
3. Are all nearby memories insufficient to produce the same complete answer?
4. Is `answer_should` correct, concise, and neither missing facts nor demanding extra facts?

Record `yes` or `no` for those four fields in the CSV, then set `overall_decision` to
`approve` or `needs_changes`. Add a concrete note whenever a field is `no`. Partial overlap is
allowed: a distractor invalidates the gold only if it independently supports the complete
expected answer.

The five cases marked **Second label required** must be reviewed independently by another
person using the `calibration` rows in the CSV. Compare labels only after both reviewers finish.

## Calibration subset

- `people-sasha-comms-training-partner`
- `preference-poppy-long-walk-morning`
- `project-coverage-pilot-expansion`
- `decision-kb-freeze-window`
- `global-branch-leads-channel`

---

## Case 1: `people-owen-escalation-contact`

- Category: `people`
- Second label required: No
- Curator-repaired: Yes
- Gold content profile: `short`
- Secondary tags: `same-entity`, `implicit-context`

### Situation

Mara is updating an internal contact guide and needs the general IT owner for systems rollouts and support workflows, not a project-specific escalation path.

### Query

> Who leads IT Services and is our general point of contact for systems rollouts and support workflows?

### Expected answer

> Identify Owen Reed as the IT Services lead and Brightgrove's general point of contact for systems rollouts and support workflows.

### Gold memory

**`mem-0021` — Owen leads IT Services and is the main rollout/support contact**

Category: `people`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

The gold states Owen's leadership role and general contact responsibility; the pilot decision only names him for a project-specific escalation path.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0001` — System and access issue escalation set for the Coverage Pilot**

Category: `decision`  
Content profile: `episodic`  
Contents: During the Shift Coverage Pilot, ownership for escalation of system or access problems was assigned to the IT Services Team. Owen Reed was named as the escalation contact for those issues. This escalation path sits alongside the pilot’s regular feedback and issue-review cadence, including weekly check-ins that involve Mara Lin, Owen Reed, and Eli Mercado to review open items and decide what needs escalation. IT Services also administers LanternDesk for the Brightgrove Public Library Network, including configuration and helpdesk reporting.

**`mem-0010` — IT Services administers LanternDesk**

Category: `global_fact`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0032` — Mara prefers async coordination with IT; call Owen only when blocked**

Category: `preference`  
Content profile: `short`  
Contents: _Fact fully captured by description._

</details>

<details>
<summary>Original generated query</summary>

> Who’s the main IT contact for systems rollouts and support workflow questions at Brightgrove?

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 2: `people-sasha-comms-training-partner`

- Category: `people`
- Second label required: **Yes**
- Curator-repaired: Yes
- Gold content profile: `medium`
- Secondary tags: `same-topic`, `paraphrase`

### Situation

Mara is writing an orientation note and wants to describe Sasha's role and how they work together without reopening old notes.

### Query

> What's Sasha's exact job title at Brightgrove, and which working group are we in together?

### Expected answer

> Identify Sasha Kim as a communications specialist and say she works closely with Mara in the Comms & Training Working Group.

### Gold memory

**`mem-0022` — Sasha Kim’s Brightgrove role and working-group connection to Mara**

Category: `people`  
Content profile: `medium`  
Contents: Sasha Kim is a communications specialist at Brightgrove. She works closely with Mara Lin as a partner in the Comms & Training Working Group. This is the main context for how they collaborate on staff communications and related work.

### Why this is intended as gold

The gold uniquely contains both Sasha's job role and the working group she shares with Mara; project memories contain only narrower collaboration details.

<details>
<summary>Closest distractors (2)</summary>

**`mem-0044` — Nina approves network-wide messages before Sasha sends them through Comms & Training**

Category: `workflow`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0041` — Sasha and Mara partner on SRO 2026 comms and training**

Category: `project`  
Content profile: `medium`  
Contents: For Summer Reading Ops 2026, Sasha Kim partners with Mara Lin through the Comms & Training Working Group. Their collaboration focuses on preparing staff-facing updates and training announcements. The working group channel is the mechanism used to coordinate these communications and training-related deliverables.

</details>

<details>
<summary>Original generated query</summary>

> Who do I usually partner with in the Comms & Training working group for staff-facing announcements?

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 3: `people-jordan-package-backup`

- Category: `people`
- Second label required: No
- Curator-repaired: Yes
- Gold content profile: `short`
- Secondary tags: `same-entity`, `implicit-context`

### Situation

Mara sees Jordan's name in a neighborhood thread and wants to place how they know each other outside occasional household favors.

### Query

> Besides package pickup, how do I know Jordan socially, and what activity do we sometimes do together?

### Expected answer

> Say Jordan Wu is Mara's neighbor and occasional running buddy.

### Gold memory

**`mem-0019` — Jordan is Mara’s neighbor, occasional running buddy, and package-pickup fallback**

Category: `people`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

Only the gold states both the neighbor relationship and their occasional runs; household memories mention Jordan only as a package fallback.

<details>
<summary>Closest distractors (2)</summary>

**`mem-0013` — When both are out, Jordan Wu is the package backup**

Category: `household_logistics`  
Content profile: `medium`  
Contents: If a delivery arrives when neither Mara nor Devon is home, Jordan Wu is the fallback contact. They text Jordan to pick up the package and hold it until they’re back. This keeps deliveries from being left unattended when the house is empty.

**`mem-0017` — Devon is Mara’s partner in the Lin–Hart household**

Category: `people`  
Content profile: `short`  
Contents: _Fact fully captured by description._

</details>

<details>
<summary>Original generated query</summary>

> If a package shows up while we’re both out, who do we usually ask to pick it up?

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 4: `preference-calendar-source-of-truth`

- Category: `preference`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `short`
- Secondary tags: `same-topic`, `paraphrase`

### Situation

After juggling several schedule changes across work and home, Mara wants to remind the assistant how to treat conflicting info from messages versus her calendar.

### Query

> When there’s a conflict between a message thread and my schedule, what should you treat as the source of truth?

### Expected answer

> A correct answer should state that Aurora Calendar should be treated as the source of truth for both work and personal scheduling.

### Gold memory

**`mem-0030` — Aurora Calendar is Mara’s scheduling source of truth**

Category: `preference`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

The gold memory directly defines the scheduling source of truth; the contexts reference calendars and planning but don’t set the preference rule.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0046` — Weekly household planning is done by updating shared Aurora Calendar events**

Category: `workflow`  
Content profile: `medium`  
Contents: In the Lin–Hart household, Mara and Devon do a weekly planning sync by updating shared events in Aurora Calendar. They use those shared events to coordinate needs for the Maple Street Townhouse and Poppy-related items. Keeping the calendar current is their standard method for aligning on home logistics.

**`mem-0043` — After meetings, due-date action items are added to Aurora Calendar**

Category: `workflow`  
Content profile: `medium`  
Contents: After meetings, Mara turns any action item that has a due date into an Aurora Calendar task or event. The point is to make the commitment appear directly on her schedule. This step applies specifically to action items with due dates.

**`mem-0007` — Brightgrove uses Aurora Calendar for scheduling and LanternDesk for operational and IT requests**

Category: `global_fact`  
Content profile: `short`  
Contents: _Fact fully captured by description._

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 5: `preference-concise-up-front-updates`

- Category: `preference`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `short`
- Secondary tags: `same-topic`, `paraphrase`

### Situation

Mara is skimming updates between meetings and wants the assistant to format a status note in her preferred style so she can read it quickly.

### Query

> When you send me project updates, how should you structure them so I can scan them fast?

### Expected answer

> A correct answer should specify that Mara prefers concise updates first (short summary and next steps), with details only if requested.

### Gold memory

**`mem-0031` — Mara wants summary and next steps first; details only when requested**

Category: `preference`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

The gold memory states Mara’s communication style preference; the contexts relate to meeting notes and recaps but don’t set the summary-first rule for updates.

<details>
<summary>Closest distractors (2)</summary>

**`mem-0050` — Mara’s Pine Notes meeting notes end with an action list labeled ‘Next steps’**

Category: `workflow`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0029` — Mara wants fast-meeting recaps written up in Pine Notes**

Category: `preference`  
Content profile: `medium`  
Contents: After fast meetings, Mara Lin prefers a written recap rather than relying on an informal verbal summary. She specifically wants decisions and action items captured in Pine Notes. This creates a clear record of what was agreed and what needs to happen next.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 6: `preference-morning-focus-time`

- Category: `preference`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `short`
- Secondary tags: `same-topic`, `implicit-context`

### Situation

Someone proposes a morning meeting slot, and Mara is trying to decide whether to accept or push it later based on her usual scheduling preference.

### Query

> Do I try to keep mornings open for focus time, or am I fine scheduling meetings then?

### Expected answer

> A correct answer should say Mara prefers to protect focus time in the mornings and schedule most meetings later in the day when possible.

### Gold memory

**`mem-0028` — Mara prefers morning focus time; meetings later when possible**

Category: `preference`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

The gold memory explicitly states the morning focus-time preference; the contexts touch on scheduling practices but don’t specify the meeting-time preference.

<details>
<summary>Closest distractors (2)</summary>

**`mem-0030` — Aurora Calendar is Mara’s scheduling source of truth**

Category: `preference`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0040` — Quarterly status check-ins are blocked on Aurora Calendar**

Category: `project`  
Content profile: `medium`  
Contents: Mara Lin keeps quarterly project status check-ins scheduled as a recurring block in Aurora Calendar. These check-ins cover the Shift Coverage Pilot, the Knowledge Base Refresh, and Summer Reading Ops 2026. The recurring calendar block serves as the standing cadence for reviewing status across these major efforts.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 7: `preference-poppy-long-walk-morning`

- Category: `preference`
- Second label required: **Yes**
- Curator-repaired: No
- Gold content profile: `episodic`
- Secondary tags: `temporal`, `same-topic`

### Situation

After a few hectic weeks, Mara is re-checking the household routine so she can plan the day and coordinate with Devon about Poppy’s exercise.

### Query

> For Poppy’s longer walk, are we doing that in the morning or after work these days?

### Expected answer

> A correct answer should say that since mid-April 2026, the household prefers Poppy’s longer walk in the morning before the day starts (not the evening).

### Gold memory

**`mem-0027` — Preferred timing for Poppy’s longer walk changed in mid-April 2026**

Category: `preference`  
Content profile: `episodic`  
Contents: Before April 15, 2026, Mara and Devon generally preferred to take Poppy’s longer walk in the evening after work. On April 15, they changed that routine so the longer walk would happen in the morning before the day starts. They carried the change into their weekly household planning sync, where they coordinate Poppy-related needs and other home logistics. Aurora Calendar remains their shared place for keeping those plans aligned, and the morning walk is now the current preference.

### Why this is intended as gold

The gold memory contains the current preference and timing change; the contexts are related pet/household details but don’t establish the long-walk timing preference.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0015` — June change to Poppy’s weekday midday-walk routine**

Category: `household_logistics`  
Content profile: `episodic`  
Contents: Earlier in 2026, Devon usually handled Poppy’s weekday midday walk because his schedule was more flexible. Starting in June 2026, the handoff changed: Mara usually takes the midday walk and, when possible, blocks a 20-minute slot for it in Aurora Calendar. Mara and Devon keep changes like this coordinated during their weekly household planning sync. They update shared Aurora Calendar events there for household and Poppy-related needs, so the new walking responsibility is reflected alongside the rest of their shared routine.

**`mem-0023` — Poppy’s vet-planning reminder is in Aurora Calendar for September 14, 2026**

Category: `plan`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0014` — Poppy’s kibble is in the pantry bin; the reorder note is in Pine Notes**

Category: `household_logistics`  
Content profile: `short`  
Contents: _Fact fully captured by description._

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 8: `project-coverage-pilot-tracking`

- Category: `project`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `medium`
- Secondary tags: `same-topic`, `implicit-context`

### Situation

A few months into the Shift Coverage Pilot, Mara is pulling together a quick update and wants to make sure the team is logging issues in the right place before she asks for a trends report.

### Query

> Where are we supposed to track requests and issues for the shift coverage pilot again?

### Expected answer

> A correct answer should specify that all Shift Coverage Pilot requests and issues are tracked as tickets in LanternDesk.

### Gold memory

**`mem-0035` — Track all Shift Coverage Pilot issues as LanternDesk tickets**

Category: `project`  
Content profile: `medium`  
Contents: For the Shift Coverage Pilot, every request and issue tied to the pilot should be logged in LanternDesk as a ticket. The purpose is to keep the work centralized so the team can report on trends and turnaround time. This applies to all pilot-related items rather than only major incidents.

### Why this is intended as gold

Only the gold memory states the required system of record (LanternDesk tickets) for pilot-related requests/issues.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0034` — Riverbend is the pilot site, with Eli as feedback contact**

Category: `project`  
Content profile: `medium`  
Contents: The Shift Coverage Pilot is primarily being run at Riverbend Branch Library. Staff feedback routes through Eli Mercado, who serves as the frontline point of contact. This establishes both the main pilot location and the person responsible for gathering and fielding input from staff during the pilot.

**`mem-0036` — Mara holds weekly Branch Leads Council check-ins during pilot**

Category: `project`  
Content profile: `medium`  
Contents: During the Shift Coverage Pilot, Mara Lin runs a weekly check-in with the Branch Leads Council. These check-ins are used to gather feedback and to confirm any process tweaks before changes are applied network-wide. The cadence provides a regular point to validate adjustments while the pilot is in progress.

**`mem-0033` — Coverage Pilot expands to HQ teams after Riverbend to test cross-site escalation**

Category: `project`  
Content profile: `short`  
Contents: _Fact fully captured by description._

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 9: `project-kb-refresh-ownership`

- Category: `project`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `short`
- Secondary tags: `same-topic`, `same-entity`

### Situation

Mara is onboarding a new contributor to the documentation effort and needs to clarify who actually owns the Knowledge Base Refresh before assigning work.

### Query

> Who owns the Knowledge Base Refresh project?

### Expected answer

> A correct answer should state that the Knowledge Base Refresh is owned by the Operations Programs Team, with Mara coordinating scope and milestones.

### Gold memory

**`mem-0039` — Operations Programs owns KB Refresh; Mara coordinates scope and milestones**

Category: `project`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

Only the gold memory explicitly defines project ownership and Mara’s coordinating role.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0037` — KB Refresh outline and draft structure are kept in Pine Notes**

Category: `project`  
Content profile: `medium`  
Contents: The working outline and draft page structure for the Knowledge Base Refresh are maintained in Pine Notes. Keeping the drafts there allows contributors to review and comment before anything is finalized. Pine Notes is the shared location for the in-progress structure rather than a later-stage publishing destination.

**`mem-0038` — Owen approves IT review for KB Refresh changes touching LanternDesk**

Category: `project`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0040` — Quarterly status check-ins are blocked on Aurora Calendar**

Category: `project`  
Content profile: `medium`  
Contents: Mara Lin keeps quarterly project status check-ins scheduled as a recurring block in Aurora Calendar. These check-ins cover the Shift Coverage Pilot, the Knowledge Base Refresh, and Summer Reading Ops 2026. The recurring calendar block serves as the standing cadence for reviewing status across these major efforts.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 10: `project-kb-refresh-it-approver`

- Category: `project`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `short`
- Secondary tags: `same-topic`, `cross-category`

### Situation

While finalizing a Knowledge Base page that touches helpdesk steps, Mara wants to route it correctly so it doesn’t get stuck in review.

### Query

> For KB Refresh changes that touch LanternDesk procedures, who’s the IT approver again?

### Expected answer

> A correct answer should name Owen Reed as the approver for the required IT Services review of KB Refresh changes that touch LanternDesk procedures.

### Gold memory

**`mem-0038` — Owen approves IT review for KB Refresh changes touching LanternDesk**

Category: `project`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

Only the gold memory ties LanternDesk-procedure-related KB Refresh changes to an IT review gate and specifies Owen as the approver.

<details>
<summary>Closest distractors (2)</summary>

**`mem-0039` — Operations Programs owns KB Refresh; Mara coordinates scope and milestones**

Category: `project`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0035` — Track all Shift Coverage Pilot issues as LanternDesk tickets**

Category: `project`  
Content profile: `medium`  
Contents: For the Shift Coverage Pilot, every request and issue tied to the pilot should be logged in LanternDesk as a ticket. The purpose is to keep the work centralized so the team can report on trends and turnaround time. This applies to all pilot-related items rather than only major incidents.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 11: `project-sro-kickoff-location`

- Category: `project`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `episodic`
- Secondary tags: `temporal`, `same-project`

### Situation

Mara is putting together a recap for someone who missed the initial alignment and wants to reference where the Summer Reading Ops work started.

### Query

> Where did we do the kickoff for Summer Reading Ops 2026?

### Expected answer

> A correct answer should say the kickoff was an in-person working session at Brightgrove HQ.

### Gold memory

**`mem-0042` — Summer Reading Ops 2026 kickoff and follow-up record**

Category: `project`  
Content profile: `episodic`  
Contents: Summer Reading Ops 2026 began with an in-person working session at Brightgrove HQ. During the kickoff, the team aligned timelines and branch dependencies. After the session, Sasha Kim and Mara Lin continued the work through the Comms & Training Working Group, preparing staff-facing updates and training announcements. That handoff placed the follow-up communication and training work with the group that coordinates those staff-facing communication and training rollouts across the Brightgrove Public Library Network.

### Why this is intended as gold

Only the gold memory states the kickoff location and that it was in-person at HQ.

<details>
<summary>Closest distractors (2)</summary>

**`mem-0041` — Sasha and Mara partner on SRO 2026 comms and training**

Category: `project`  
Content profile: `medium`  
Contents: For Summer Reading Ops 2026, Sasha Kim partners with Mara Lin through the Comms & Training Working Group. Their collaboration focuses on preparing staff-facing updates and training announcements. The working group channel is the mechanism used to coordinate these communications and training-related deliverables.

**`mem-0040` — Quarterly status check-ins are blocked on Aurora Calendar**

Category: `project`  
Content profile: `medium`  
Contents: Mara Lin keeps quarterly project status check-ins scheduled as a recurring block in Aurora Calendar. These check-ins cover the Shift Coverage Pilot, the Knowledge Base Refresh, and Summer Reading Ops 2026. The recurring calendar block serves as the standing cadence for reviewing status across these major efforts.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 12: `project-coverage-pilot-expansion`

- Category: `project`
- Second label required: **Yes**
- Curator-repaired: Yes
- Gold content profile: `short`
- Secondary tags: `same-project`, `temporal`, `cross-category`, `two-part-question`

### Situation

After the Riverbend pilot period, Mara is briefing IT and the HQ team leads on the next phase and wants to confirm both its audience and purpose.

### Query

> After the Riverbend pilot period, where is the Shift Coverage Pilot supposed to expand next, and what will that phase test?

### Expected answer

> Say the pilot is supposed to expand to HQ-based teams and that the next phase will test cross-site escalation paths.

### Gold memory

**`mem-0033` — Coverage Pilot expands to HQ teams after Riverbend to test cross-site escalation**

Category: `project`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

Only the gold describes the post-Riverbend expansion to HQ-based teams and its cross-site escalation purpose; nearby memories cover the current site, the kickoff location, or the existing escalation owner.

<details>
<summary>Closest distractors (4)</summary>

**`mem-0034` — Riverbend is the pilot site, with Eli as feedback contact**

Category: `project`  
Content profile: `medium`  
Contents: The Shift Coverage Pilot is primarily being run at Riverbend Branch Library. Staff feedback routes through Eli Mercado, who serves as the frontline point of contact. This establishes both the main pilot location and the person responsible for gathering and fielding input from staff during the pilot.

**`mem-0053` — Riverbend selected to launch the Shift Coverage Pilot first**

Category: `decision`  
Content profile: `episodic`  
Contents: Mara Lin and Nina Patel selected Riverbend Branch Library as the first site for the Shift Coverage Pilot. They chose Riverbend because it has consistent supervisor coverage and because there was a reliable feedback loop through Eli Mercado. After the selection, Riverbend was established as the pilot’s primary site, with Eli serving as the frontline point of contact for staff feedback. The pilot also used a broader feedback structure, including Mara’s weekly check-in with the Branch Leads Council to gather feedback and confirm any process tweaks before they go network-wide. Mara later planned an on-site visit to Riverbend Branch Library on 2026-04-23 to gather frontline feedback for the pilot with Eli.

**`mem-0058` — Coverage Pilot kickoff is March 19 at HQ; Mara leads and Nina handles approvals**

Category: `plan`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0001` — System and access issue escalation set for the Coverage Pilot**

Category: `decision`  
Content profile: `episodic`  
Contents: During the Shift Coverage Pilot, ownership for escalation of system or access problems was assigned to the IT Services Team. Owen Reed was named as the escalation contact for those issues. This escalation path sits alongside the pilot’s regular feedback and issue-review cadence, including weekly check-ins that involve Mara Lin, Owen Reed, and Eli Mercado to review open items and decide what needs escalation. IT Services also administers LanternDesk for the Brightgrove Public Library Network, including configuration and helpdesk reporting.

</details>

<details>
<summary>Original generated query</summary>

> Which branch is our main site for the shift coverage pilot, and who’s the frontline contact there?

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 13: `decision-kb-freeze-window`

- Category: `decision`
- Second label required: **Yes**
- Curator-repaired: Yes
- Gold content profile: `episodic`
- Secondary tags: `temporal`, `same-project`

### Situation

Mara already has the approved late-April freeze dates in front of her, but needs to explain Nina's reason for the decision to contributors.

### Query

> Why did Nina approve the Knowledge Base Refresh documentation freeze from April 20 through April 30?

### Expected answer

> Explain that Nina approved the freeze to allow a clean review pass and avoid midstream edits.

### Gold memory

**`mem-0003` — Documentation freeze set for the Knowledge Base Refresh review window**

Category: `decision`  
Content profile: `episodic`  
Contents: Nina Patel approved a documentation freeze for the Knowledge Base Refresh covering 2026-04-20 through 2026-04-30. The purpose of the freeze was to let the team complete a clean review pass and avoid midstream edits while that review was underway. In the broader refresh workflow, the working outline and draft page structure were kept in Pine Notes so contributors could comment before anything was finalized. Separately, Mara Lin had also blocked 2026-04-06 through 2026-04-10 on Aurora Calendar as a documentation freeze window to finish the outline and page inventory.

### Why this is intended as gold

Only the gold records Nina's approved decision and its rationale; Mara's earlier personal calendar block was for finishing the outline and page inventory.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0037` — KB Refresh outline and draft structure are kept in Pine Notes**

Category: `project`  
Content profile: `medium`  
Contents: The working outline and draft page structure for the Knowledge Base Refresh are maintained in Pine Notes. Keeping the drafts there allows contributors to review and comment before anything is finalized. Pine Notes is the shared location for the in-progress structure rather than a later-stage publishing destination.

**`mem-0039` — Operations Programs owns KB Refresh; Mara coordinates scope and milestones**

Category: `project`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0024` — Documentation freeze block set for Knowledge Base Refresh deliverables**

Category: `plan`  
Content profile: `medium`  
Contents: Mara blocked 2026-04-06 through 2026-04-10 on Aurora Calendar as a documentation freeze window. The purpose of that block was to finish the Knowledge Base Refresh outline and complete the page inventory. Keeping it on the calendar protected time specifically for those documentation tasks.

</details>

<details>
<summary>Original generated query</summary>

> What were the exact dates for the KB Refresh documentation freeze we agreed on?

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 14: `decision-sro-training-format-current`

- Category: `decision`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `episodic`
- Secondary tags: `temporal`, `supersession`, `same-project`

### Situation

As Summer Reading Ops gets closer, Mara is confirming the training logistics with Sasha and needs to restate the final format choice to include in a staff update.

### Query

> What format did we end up choosing for the Summer Reading Ops 2026 staff training?

### Expected answer

> A correct answer should state the final decision: a hybrid training plan with a shorter in-person session at Brightgrove HQ plus a recorded follow-up distributed by the Comms & Training Working Group.

### Gold memory

**`mem-0004` — Summer Reading Ops 2026 training-format decision record**

Category: `decision`  
Content profile: `episodic`  
Contents: For Summer Reading Ops 2026, Mara Lin changed the staff training approach from the earlier plan. The final format became a hybrid: a shorter in-person session at Brightgrove HQ paired with a recorded follow-up. The recorded component was to be distributed by the Comms & Training Working Group. This fit the working group’s remit of coordinating staff-facing announcements and training rollouts for Brightgrove Public Library Network initiatives, and it leveraged HQ as the network’s primary in-person site when teams choose to meet on-site.

### Why this is intended as gold

Only the gold memory reflects the current, non-superseded training format decision and includes both components and the distribution path.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0005` — Initial SRO 2026 staff training plan was an HQ-only session**

Category: `decision`  
Content profile: `medium`  
Contents: Mara Lin initially chose to run Summer Reading Ops 2026 staff training as an in-person session at Brightgrove HQ. The intent was to reduce confusion during rollout by keeping training together on-site. This plan was later superseded by a different training format.

**`mem-0041` — Sasha and Mara partner on SRO 2026 comms and training**

Category: `project`  
Content profile: `medium`  
Contents: For Summer Reading Ops 2026, Sasha Kim partners with Mara Lin through the Comms & Training Working Group. Their collaboration focuses on preparing staff-facing updates and training announcements. The working group channel is the mechanism used to coordinate these communications and training-related deliverables.

**`mem-0042` — Summer Reading Ops 2026 kickoff and follow-up record**

Category: `project`  
Content profile: `episodic`  
Contents: Summer Reading Ops 2026 began with an in-person working session at Brightgrove HQ. During the kickoff, the team aligned timelines and branch dependencies. After the session, Sasha Kim and Mara Lin continued the work through the Comms & Training Working Group, preparing staff-facing updates and training announcements. That handoff placed the follow-up communication and training work with the group that coordinates those staff-facing communication and training rollouts across the Brightgrove Public Library Network.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 15: `decision-coverage-escalation-owner`

- Category: `decision`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `episodic`
- Secondary tags: `same-project`, `routing`

### Situation

A system access problem comes up during the Shift Coverage Pilot, and Mara wants to route it correctly without re-litigating the escalation path.

### Query

> For the shift coverage pilot, who did we assign escalation to for system or access issues?

### Expected answer

> A correct answer should say escalation for system/access problems goes to the IT Services Team, with Owen Reed as the escalation contact.

### Gold memory

**`mem-0001` — System and access issue escalation set for the Coverage Pilot**

Category: `decision`  
Content profile: `episodic`  
Contents: During the Shift Coverage Pilot, ownership for escalation of system or access problems was assigned to the IT Services Team. Owen Reed was named as the escalation contact for those issues. This escalation path sits alongside the pilot’s regular feedback and issue-review cadence, including weekly check-ins that involve Mara Lin, Owen Reed, and Eli Mercado to review open items and decide what needs escalation. IT Services also administers LanternDesk for the Brightgrove Public Library Network, including configuration and helpdesk reporting.

### Why this is intended as gold

Only the gold memory states the explicit escalation ownership decision and names the escalation contact.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0035` — Track all Shift Coverage Pilot issues as LanternDesk tickets**

Category: `project`  
Content profile: `medium`  
Contents: For the Shift Coverage Pilot, every request and issue tied to the pilot should be logged in LanternDesk as a ticket. The purpose is to keep the work centralized so the team can report on trends and turnaround time. This applies to all pilot-related items rather than only major incidents.

**`mem-0045` — Shift Coverage Pilot issues are triaged weekly with Owen and Eli**

Category: `workflow`  
Content profile: `medium`  
Contents: During the Shift Coverage Pilot, Mara runs a weekly check-in with Owen Reed and Eli Mercado. In that meeting, they review open issues and decide what needs escalation. The weekly cadence is used to keep triage and escalation decisions moving during the pilot.

**`mem-0002` — Coverage Pilot feedback was routed through the Branch Leads Council**

Category: `decision`  
Content profile: `episodic`  
Contents: A decision was made to treat the Branch Leads Council as the official channel for Shift Coverage Pilot feedback. As a result, ad hoc emails about the pilot were redirected into the council’s weekly discussion rather than handled in parallel threads. This aligned with the existing practice during the pilot where Mara Lin runs a weekly check-in with the Branch Leads Council to gather feedback and confirm process tweaks before anything goes network-wide. Eli Mercado participates as a regular representative in the Branch Leads Council for frontline feedback.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 16: `global-tools-standards`

- Category: `global_fact`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `short`
- Secondary tags: `same-topic`, `cross-category`, `paraphrase`

### Situation

Months after onboarding a new coordinator, Mara is drafting a quick orientation note and wants to point them to the right systems for scheduling and request tracking.

### Query

> What are our standard tools again for scheduling and for tracking ops/IT requests across Brightgrove?

### Expected answer

> A correct answer should name Aurora Calendar as the standard scheduling tool and LanternDesk as the standard system for tracking operational and IT requests across Brightgrove.

### Gold memory

**`mem-0007` — Brightgrove uses Aurora Calendar for scheduling and LanternDesk for operational and IT requests**

Category: `global_fact`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

The gold memory explicitly states both network-wide standards (scheduling and request tracking); the context memories mention related tools/teams but do not establish the cross-network standards pairing.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0010` — IT Services administers LanternDesk**

Category: `global_fact`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0011` — Pine Notes is commonly used for internal drafts**

Category: `global_fact`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0009` — HQ is Brightgrove’s main in-person meeting/training site**

Category: `global_fact`  
Content profile: `short`  
Contents: _Fact fully captured by description._

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 17: `global-branch-leads-channel`

- Category: `global_fact`
- Second label required: **Yes**
- Curator-repaired: Yes
- Gold content profile: `medium`
- Secondary tags: `same-entity`, `same-topic`, `paraphrase`

### Situation

A new project lead knows the Branch Leads Council handles some pilot feedback but does not understand the council's permanent network-wide role.

### Query

> Outside individual projects, what is the Branch Leads Council's standing role at Brightgrove?

### Expected answer

> Explain that the Branch Leads Council is Brightgrove's standing channel for frontline feedback on network-wide operational changes before broad rollout.

### Gold memory

**`mem-0006` — Branch Leads Council provides frontline input before network-wide rollouts**

Category: `global_fact`  
Content profile: `medium`  
Contents: The Branch Leads Council serves as Brightgrove’s standing channel for frontline feedback on network-wide operational changes. Its role is to gather and surface input before changes are rolled out broadly across the network. This makes it the default forum for operational feedback outside any single project’s specific communication plan.

### Why this is intended as gold

The gold defines the council's permanent network-wide role and when that feedback is used; the pilot decision only assigns it as the channel for one project.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0018` — Eli is a branch supervisor and regular Branch Leads Council representative**

Category: `people`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0002` — Coverage Pilot feedback was routed through the Branch Leads Council**

Category: `decision`  
Content profile: `episodic`  
Contents: A decision was made to treat the Branch Leads Council as the official channel for Shift Coverage Pilot feedback. As a result, ad hoc emails about the pilot were redirected into the council’s weekly discussion rather than handled in parallel threads. This aligned with the existing practice during the pilot where Mara Lin runs a weekly check-in with the Branch Leads Council to gather feedback and confirm process tweaks before anything goes network-wide. Eli Mercado participates as a regular representative in the Branch Leads Council for frontline feedback.

**`mem-0036` — Mara holds weekly Branch Leads Council check-ins during pilot**

Category: `project`  
Content profile: `medium`  
Contents: During the Shift Coverage Pilot, Mara Lin runs a weekly check-in with the Branch Leads Council. These check-ins are used to gather feedback and to confirm any process tweaks before changes are applied network-wide. The cadence provides a regular point to validate adjustments while the pilot is in progress.

</details>

<details>
<summary>Original generated query</summary>

> Which group is our standing channel for frontline feedback on network-wide operational changes?

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 18: `workflow-notes-next-steps`

- Category: `workflow`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `short`
- Secondary tags: `paraphrase`, `same-tool`, `implicit-context`

### Situation

Weeks after a busy stretch of meetings, Mara is setting up a fresh Pine Notes template and wants it to match how she normally closes out meeting notes so she can scan them quickly later.

### Query

> When I take meeting notes, what’s the label I usually use at the end for the action-items list?

### Expected answer

> A correct answer should say that Mara ends meeting notes with a short action-items list labeled "Next steps" (and that she captures the notes in Pine Notes).

### Gold memory

**`mem-0050` — Mara’s Pine Notes meeting notes end with an action list labeled ‘Next steps’**

Category: `workflow`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

Only the gold memory specifies the exact label used at the end of Mara’s meeting notes; related workflow memories cover tasks and recaps but not that label.

<details>
<summary>Closest distractors (2)</summary>

**`mem-0043` — After meetings, due-date action items are added to Aurora Calendar**

Category: `workflow`  
Content profile: `medium`  
Contents: After meetings, Mara turns any action item that has a due date into an Aurora Calendar task or event. The point is to make the commitment appear directly on her schedule. This step applies specifically to action items with due dates.

**`mem-0029` — Mara wants fast-meeting recaps written up in Pine Notes**

Category: `preference`  
Content profile: `medium`  
Contents: After fast meetings, Mara Lin prefers a written recap rather than relying on an informal verbal summary. She specifically wants decisions and action items captured in Pine Notes. This creates a clear record of what was agreed and what needs to happen next.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 19: `workflow-route-requests`

- Category: `workflow`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `medium`
- Secondary tags: `same-tool`, `same-topic`, `paraphrase`

### Situation

A colleague pings Mara with a request and asks if email is fine. Mara wants to respond with the standard way she handles these so the request has an owner and history.

### Query

> What’s my usual process for operational or IT requests—do I handle them over email or somewhere else?

### Expected answer

> A correct answer should state that Mara routes operational or IT requests through LanternDesk rather than email, so they have an owner, status, and history.

### Gold memory

**`mem-0049` — Operational and IT requests go through LanternDesk rather than email**

Category: `workflow`  
Content profile: `medium`  
Contents: Mara routes operational or IT requests through LanternDesk instead of handling them over email. She uses LanternDesk so each request has an owner, a status, and a history. This is her usual process for tracking and managing those requests.

### Why this is intended as gold

The gold memory directly defines Mara’s request-routing workflow (LanternDesk instead of email) and the rationale; context memories are adjacent but don’t establish the general rule.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0010` — IT Services administers LanternDesk**

Category: `global_fact`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0035` — Track all Shift Coverage Pilot issues as LanternDesk tickets**

Category: `project`  
Content profile: `medium`  
Contents: For the Shift Coverage Pilot, every request and issue tied to the pilot should be logged in LanternDesk as a ticket. The purpose is to keep the work centralized so the team can report on trends and turnaround time. This applies to all pilot-related items rather than only major incidents.

**`mem-0032` — Mara prefers async coordination with IT; call Owen only when blocked**

Category: `preference`  
Content profile: `short`  
Contents: _Fact fully captured by description._

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 20: `workflow-comms-approval`

- Category: `workflow`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `short`
- Secondary tags: `same-entity`, `same-topic`, `implicit-context`

### Situation

After drafting a network-wide update with Sasha, Mara is about to ask Sasha to send it out but pauses to confirm the approval step she usually follows.

### Query

> Before Sasha sends a network-wide announcement, who do I need to get sign-off from?

### Expected answer

> A correct answer should say that Mara gets Nina Patel’s sign-off before Sasha sends network-wide communications through the Comms & Training Working Group.

### Gold memory

**`mem-0044` — Nina approves network-wide messages before Sasha sends them through Comms & Training**

Category: `workflow`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

The gold memory states the specific approval gate and the people involved; context memories establish roles and the working group’s remit but don’t specify the sign-off requirement.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0020` — Nina is Mara’s manager at Brightgrove**

Category: `people`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0022` — Sasha Kim’s Brightgrove role and working-group connection to Mara**

Category: `people`  
Content profile: `medium`  
Contents: Sasha Kim is a communications specialist at Brightgrove. She works closely with Mara Lin as a partner in the Comms & Training Working Group. This is the main context for how they collaborate on staff communications and related work.

**`mem-0008` — Comms & Training WG coordinates announcements and training rollouts**

Category: `global_fact`  
Content profile: `short`  
Contents: _Fact fully captured by description._

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 21: `workflow-kb-intake-current`

- Category: `workflow`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `episodic`
- Secondary tags: `temporal`, `same-project`, `same-tool`

### Situation

A few months after the Knowledge Base Refresh process changed, a staff member asks Mara where to submit a small page fix. Mara wants to point them to the current intake path without mixing it up with the earlier ticket-based approach.

### Query

> For Knowledge Base Refresh edits now, where do I want people to submit page-fix requests, and when do we use a ticket?

### Expected answer

> A correct answer should specify that starting 2026-05-01, KB Refresh intake is via a shared Pine Notes checklist, and only escalations or access problems should get a LanternDesk ticket.

### Gold memory

**`mem-0048` — May change to the Knowledge Base Refresh intake process**

Category: `workflow`  
Content profile: `episodic`  
Contents: From 2026-01-01 through 2026-04-30, Knowledge Base Refresh page-fix requests were collected by asking staff to open a LanternDesk ticket tagged for the Knowledge Base Refresh. Starting 2026-05-01, Mara switched the intake process to a shared Pine Notes checklist for routine page-fix submissions. After this change, LanternDesk tickets are used only for escalations or access problems. The working outline and draft page structure for the Knowledge Base Refresh live in Pine Notes so contributors can comment before anything is finalized. Pine Notes is also commonly used at Brightgrove for drafting and organizing internal documentation before it is shared more broadly.

### Why this is intended as gold

The gold memory uniquely captures the current (post-2026-05-01) intake workflow and the exception for when to open a LanternDesk ticket; context memories are either older, about where drafts live, or about general request routing.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0047` — KB Refresh page-fix intake used LanternDesk tickets through April 2026**

Category: `workflow`  
Content profile: `medium`  
Contents: From 2026-01-01 through 2026-04-30, Mara collected Knowledge Base Refresh page-fix requests via LanternDesk. Staff were asked to open a LanternDesk ticket and tag it for the Knowledge Base Refresh. That ticket-based approach was the intake method throughout this date range.

**`mem-0037` — KB Refresh outline and draft structure are kept in Pine Notes**

Category: `project`  
Content profile: `medium`  
Contents: The working outline and draft page structure for the Knowledge Base Refresh are maintained in Pine Notes. Keeping the drafts there allows contributors to review and comment before anything is finalized. Pine Notes is the shared location for the in-progress structure rather than a later-stage publishing destination.

**`mem-0049` — Operational and IT requests go through LanternDesk rather than email**

Category: `workflow`  
Content profile: `medium`  
Contents: Mara routes operational or IT requests through LanternDesk instead of handling them over email. She uses LanternDesk so each request has an owner, a status, and a history. This is her usual process for tracking and managing those requests.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 22: `household-trash-night`

- Category: `household_logistics`
- Second label required: No
- Curator-repaired: Yes
- Gold content profile: `medium`
- Secondary tags: `same-topic`, `implicit-context`, `temporal`

### Situation

Mara and Devon are coordinating household responsibilities and want to confirm the standing trash and recycling routine.

### Query

> What night do trash and recycling go out, and who handles the bins by default?

### Expected answer

> A correct answer should state that trash and recycling go out Tuesday nights at the Maple Street Townhouse and that whoever is home last handles putting them out by default.

### Gold memory

**`mem-0016` — Tuesday nights are bin night, handled by whoever’s home last**

Category: `household_logistics`  
Content profile: `medium`  
Contents: At the Maple Street Townhouse, trash and recycling are put out on Tuesday nights. By default, whoever is home last is the one who takes the bins out. This rule covers the routine without needing to assign it in advance.

### Why this is intended as gold

Only the gold memory specifies both the correct night (Tuesday) and the household default rule for who puts the bins out.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0046` — Weekly household planning is done by updating shared Aurora Calendar events**

Category: `workflow`  
Content profile: `medium`  
Contents: In the Lin–Hart household, Mara and Devon do a weekly planning sync by updating shared events in Aurora Calendar. They use those shared events to coordinate needs for the Maple Street Townhouse and Poppy-related items. Keeping the calendar current is their standard method for aligning on home logistics.

**`mem-0012` — The shared grocery list is pinned in Pine Notes for Mara and Devon**

Category: `household_logistics`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0013` — When both are out, Jordan Wu is the package backup**

Category: `household_logistics`  
Content profile: `medium`  
Contents: If a delivery arrives when neither Mara nor Devon is home, Jordan Wu is the fallback contact. They text Jordan to pick up the package and hold it until they’re back. This keeps deliveries from being left unattended when the house is empty.

</details>

<details>
<summary>Original generated query</summary>

> Is tonight our trash/recycling night, and who’s supposed to put the bins out if we’re both coming home late?

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 23: `household-poppy-midday-walk`

- Category: `household_logistics`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `episodic`
- Secondary tags: `temporal`, `same-entity`, `implicit-context`

### Situation

In mid-summer, Mara is booking a midday appointment and wants to avoid colliding with the dog-walk slot. She can’t remember whether she or Devon is currently the one who usually does the weekday midday walk.

### Query

> Remind me—who’s been handling Poppy’s weekday midday walk lately, and how long do we block for it?

### Expected answer

> A correct answer should say that from June 2026 onward Mara usually handles Poppy’s weekday midday walk and that she blocks a 20-minute slot on Aurora Calendar for it.

### Gold memory

**`mem-0015` — June change to Poppy’s weekday midday-walk routine**

Category: `household_logistics`  
Content profile: `episodic`  
Contents: Earlier in 2026, Devon usually handled Poppy’s weekday midday walk because his schedule was more flexible. Starting in June 2026, the handoff changed: Mara usually takes the midday walk and, when possible, blocks a 20-minute slot for it in Aurora Calendar. Mara and Devon keep changes like this coordinated during their weekly household planning sync. They update shared Aurora Calendar events there for household and Poppy-related needs, so the new walking responsibility is reflected alongside the rest of their shared routine.

### Why this is intended as gold

Only the gold memory includes the current owner (Mara, starting June 2026) and the 20-minute calendar block duration.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0027` — Preferred timing for Poppy’s longer walk changed in mid-April 2026**

Category: `preference`  
Content profile: `episodic`  
Contents: Before April 15, 2026, Mara and Devon generally preferred to take Poppy’s longer walk in the evening after work. On April 15, they changed that routine so the longer walk would happen in the morning before the day starts. They carried the change into their weekly household planning sync, where they coordinate Poppy-related needs and other home logistics. Aurora Calendar remains their shared place for keeping those plans aligned, and the morning walk is now the current preference.

**`mem-0023` — Poppy’s vet-planning reminder is in Aurora Calendar for September 14, 2026**

Category: `plan`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0014` — Poppy’s kibble is in the pantry bin; the reorder note is in Pine Notes**

Category: `household_logistics`  
Content profile: `short`  
Contents: _Fact fully captured by description._

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 24: `plan-sro-staffing-guidance-deadline`

- Category: `plan`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `short`
- Secondary tags: `same-project`, `temporal`, `paraphrase`

### Situation

Near the start of May, Mara is lining up the Comms & Training Working Group agenda and wants to confirm the target date she set for having the Summer Reading staffing guidance draft ready for review.

### Query

> When did I say I’d have the summer reading staffing guidance draft ready for Comms & Training to review?

### Expected answer

> A correct answer should give the target date of 2026-05-08 and specify it’s the staffing guidance draft for Summer Reading Ops 2026 for Comms & Training Working Group review.

### Gold memory

**`mem-0026` — Summer Reading Ops staffing-guidance draft is due May 8 for Comms & Training review**

Category: `plan`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

Only the gold memory contains the specific planned deadline date and what it’s for (staffing guidance draft for Comms & Training review).

<details>
<summary>Closest distractors (3)</summary>

**`mem-0041` — Sasha and Mara partner on SRO 2026 comms and training**

Category: `project`  
Content profile: `medium`  
Contents: For Summer Reading Ops 2026, Sasha Kim partners with Mara Lin through the Comms & Training Working Group. Their collaboration focuses on preparing staff-facing updates and training announcements. The working group channel is the mechanism used to coordinate these communications and training-related deliverables.

**`mem-0042` — Summer Reading Ops 2026 kickoff and follow-up record**

Category: `project`  
Content profile: `episodic`  
Contents: Summer Reading Ops 2026 began with an in-person working session at Brightgrove HQ. During the kickoff, the team aligned timelines and branch dependencies. After the session, Sasha Kim and Mara Lin continued the work through the Comms & Training Working Group, preparing staff-facing updates and training announcements. That handoff placed the follow-up communication and training work with the group that coordinates those staff-facing communication and training rollouts across the Brightgrove Public Library Network.

**`mem-0004` — Summer Reading Ops 2026 training-format decision record**

Category: `decision`  
Content profile: `episodic`  
Contents: For Summer Reading Ops 2026, Mara Lin changed the staff training approach from the earlier plan. The final format became a hybrid: a shorter in-person session at Brightgrove HQ paired with a recorded follow-up. The recorded component was to be distributed by the Comms & Training Working Group. This fit the working group’s remit of coordinating staff-facing announcements and training rollouts for Brightgrove Public Library Network initiatives, and it leveraged HQ as the network’s primary in-person site when teams choose to meet on-site.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---

## Case 25: `plan-ops-team-day`

- Category: `plan`
- Second label required: No
- Curator-repaired: No
- Gold content profile: `short`
- Secondary tags: `same-team`, `temporal`, `implicit-context`

### Situation

In early June, Mara is trying to schedule a vendor call and wants to avoid a day that’s already earmarked for an in-person team day at HQ. She can’t recall the exact date.

### Query

> What date is our Ops Programs in-person team day at HQ?

### Expected answer

> A correct answer should state that the Operations Programs Team’s in-person team day at Brightgrove HQ is on 2026-06-11 (and note it’s kept meeting-light in Aurora Calendar if mentioned).

### Gold memory

**`mem-0025` — Ops Programs team day is June 11 at HQ; Mara keeps it meeting-light**

Category: `plan`  
Content profile: `short`  
Contents: _Fact fully captured by description._

### Why this is intended as gold

Only the gold memory provides the specific date for the Ops Programs team day at HQ.

<details>
<summary>Closest distractors (3)</summary>

**`mem-0028` — Mara prefers morning focus time; meetings later when possible**

Category: `preference`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0009` — HQ is Brightgrove’s main in-person meeting/training site**

Category: `global_fact`  
Content profile: `short`  
Contents: _Fact fully captured by description._

**`mem-0040` — Quarterly status check-ins are blocked on Aurora Calendar**

Category: `project`  
Content profile: `medium`  
Contents: Mara Lin keeps quarterly project status check-ins scheduled as a recurring block in Aurora Calendar. These check-ins cover the Shift Coverage Pilot, the Knowledge Base Refresh, and Summer Reading Ops 2026. The recurring calendar block serves as the standing cadence for reviewing status across these major efforts.

</details>

### Human decision

- [ ] Query is natural and unambiguous
- [ ] Gold alone supports the complete expected answer
- [ ] No distractor is equally sufficient
- [ ] Expected answer is correct and appropriately scoped
- [ ] **Approve**
- [ ] **Needs changes**

Notes:

<!-- Add specific concerns or proposed corrections here. -->

---
