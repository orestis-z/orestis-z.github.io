---
layout: post
title: "Automating a Beach Rental Store"
date: 2025-12-08 12:00:00 +0100
categories: [IoT, Entrepreneurship]
tags: [quantization, pruning, model-compression, deep-learning, inference-optimization]
author: Orestis Zambounis
image: /assets/images/blogs/2025-12-08-automating-beach-rental-store/beachin.jpg
description: "A deep dive into automating a physical retail space: How I built custom smart racks using Raspberry Pi, I/O expanders, and 3D printed parts to run a 24/7 unstaffed rental store."
---

This is the story of **Beachin' Rentals**, our first venture into the world of physical retail. Located just three minutes from the sand in Barceloneta, we set out to provide high-quality beach equipment to locals and tourists alike. However, we quickly ran into a fundamental problem: the cost of staffing. In our first year, the expense of having an employee on-site was eating through our margins, and we were losing money.

We faced a stark choice: close up shop or pivot entirely. I began analyzing the daily tasks of our staff and realized the job was relatively repetitive and procedural. This sparked a question: *What if we could automate it?* Theoretically, automation would allow us to operate 24/7 with near-zero marginal labor costs, transforming a struggling rental shop into a scalable, self-sufficient business.

## Automation: The "Smart Rack"

The first hurdle was the hardware. Our core products—beach loungers and umbrellas—do not fit into conventional smart lockers due to their "oversized" geometry. Even if we could find large enough lockers, they wouldn't be space-efficient for our small storefront. We needed a custom solution.

I designed a "Smart Rack." Instead of traditional locker doors, this system uses chained buckles that loop through the product to secure it. It was a space-saving, geometry-agnostic solution perfectly detailed for long, narrow items like umbrellas and folded sunbeds.

To validate this idea, I started with a Proof of Concept (PoC). The goal was to prove the entire loop could function without human intervention: from a customer selecting an item on a UI, to processing a payment via a Stripe terminal, to triggering the electric locks. If I could make this work for one lock, I knew I could scale it to 32.

<div class="vimeo">
  <iframe src="https://player.vimeo.com/video/1144524032?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479&amp;autoplay=1&amp;muted=1&amp;loop=1&amp;controls=0" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share" referrerpolicy="strict-origin-when-cross-origin" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="Admin Panel -> Lock Control"></iframe>
</div>
<p class="image-caption">Proof of concept: Electric lock control from a separate user interface.</p>

Mounting the locks onto the rack was easier said than done. I spent roughly two months just cutting metal to fit the rack structure so the buckles would slide through correctly. There were countless hours spent 3D printing handheld parts, connecting cables, and soldering components to ensure a robust build.

<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/wip-3.jpg"
    alt="Work-in-progress metal rack frame with electric locks mounted and 3D printed handheld components attached"
/>
<p class="image-caption">Electric locks mounted on rack frame with 3D printed handheld parts.</p>

Once the electric lockers were installed, I wired up all the trigger and feedback cables using four I/O expanders (MCP23017). This setup allowed a single Raspberry Pi to control the state of 32 lockers via the I2C interface. Crucially, I implemented a feedback loop: any activity in the shop (opening or closing a lock) triggered a notification on our Android admin app. This admin interface was connected via the cloud, giving us the power to control the locks remotely if needed.

<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/wip-4.jpg"
    alt="Raspberry Pi connected to four MCP23017 I/O expanders via I2C interface with wiring setup to control 32 electric locks"
/>
<p class="image-caption">I/O expanders (MCP23017) to control 32 locks through the I2C interface of the Raspberry Pi.</p>

For the customer experience, we designed an intuitive self-service UI. We tested this rigorously with various users before rolling it out to ensure it was foolproof. The system accommodated two flows: walk-in customers could select and pay directly at the kiosk, while those who booked online via our Shopify store could enter a reservation code to unlock their gear. We opted for a physical payment terminal rather than a QR-code-based web payment to reduce trust friction and make the transaction feel more secure and professional.

<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/kiosk2b.jpg"
    alt="Self-service kiosk with touchscreen interface for product selection and integrated payment terminal for rental equipment"
/>
<p class="image-caption">User interface to select and pay for rental equipment, and unlock online orders.</p>

After more than three months of grueling work—much harder than anticipated—the system was finally live. Watching our first customers use it was surreal. We had feared that removing the human element might lower our conversion rate, but we observed the exact opposite. Customers loved the autonomy and speed of the smooth check-in/check-out process, free from the pressure of interacting with sales staff.

<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/rack.jpg"
    style="max-width:min(100%, 400px)"
    alt="Completed smart rack system with beach loungers and umbrellas secured by automated electric locks and buckle chains"
/>
<p class="image-caption">Final "Smart Rack" with rental products snugly fitted.</p>

<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/customers.jpg"
    alt="Two customers entering the self-service store Beachin' Rentals"
    style="margin-bottom: -15px"
/>
<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/customers2.jpg"
    alt="Multiple customers browsing and using the automated rental system to select beach equipment"
/>
<p class="image-caption">Customers using the self-service system to rent beach equipment.</p>

## Smart Lockers for Inline Skates

Having proven the business model was viable in our second year with just the 32-lock rack, we looked for ways to expand. We still had available floor space, and as an inline skating enthusiast myself, I knew Barcelona is considered a European mecca for the sport, thanks to its smooth, flat promenades like the Barceloneta seaside.

We launched a "smoke test" with just two pairs of inline skates to validate the demand. The results were positive, so we decided to scale up. We ordered customized smart lockers directly from a factory in China, designed to integrate easily into our existing ecosystem via the RS485 protocol.

<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/smartlockers.jpg"
    alt="Custom smart lockers manufactured in China with RS485 protocol integration designed for automated inline skate rentals"
/>
<p class="image-caption">Customized smart lockers from a factory in China.</p>

<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/smartlockerslight.jpg"
    alt="Illuminated smart lockers with LED lighting displaying inline skates in individual transparent compartments for optimal product presentation"
/>
<p class="image-caption">Smart lockers with inline skates integrated into our store. Illuminated compartments ensure optimal product presentation.</p>

The addition made the shop feel complete. The inline skates blended perfectly with our beach equipment, adding a cool, active vibe to the store while diversifying our revenue streams.

<div class="vimeo">
  <iframe src="https://player.vimeo.com/video/1144527759?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479&amp;autoplay=1&amp;muted=1&amp;loop=1&amp;controls=0" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share" referrerpolicy="strict-origin-when-cross-origin" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="Bechin' Rentals Self-Service"></iframe>
</div>
<p class="image-caption">Round view through our shop with both the smart rack and smart lockers.</p>

## Door Automation

While the rentals were automated, the store operations weren't fully hands-off yet. We still needed someone to visit the shop every evening for about 15 minutes to check returned inventory for damage and close the door blinds. Finding someone reliable for such a short, specific task proved to be a pain.

We realized that since our inventory was relatively low-cost and damage was rare, it was more economical to absorb the occasional loss of a deposit than to pay for daily manual inspections. However, we still needed to secure the shop at night to prevent vandalism. The solution was full door automation. We outsourced this to a specialized company that automated our blinds for a relatively low price point, finally removing the last daily manual task.

<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/door-fix.gif"
    alt="Technicians working on installing automated door blind system to enable hands-free store opening and closing"
/>
<p class="image-caption">Technicians working on fully automating our door blinds.</p>

## Upgraded Rack (V2)

Our first smart rack was a success, often selling out completely during the high season. However, it had flaws: the beach umbrellas were simply placed on a shelf, leading to tangles and mess. We decided to almost double our capacity from 32 to 62 locks with an upgraded design.

This time, I didn't want to spend weeks DIY-ing the fabrication. We utilized an on-demand laser-cutting service in Germany to create modular metal pieces that could be mounted onto any standard rack to convert it into a smart system with minimal effort. We also 3D printed several components on-demand in China, as our design had reached a level of maturity that justified outsourcing.

<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/module.jpg"
    alt="Precision laser-cut modular metal components designed for easy smart rack assembly and scalability"
    style="max-width:min(100%, 400px)"
/>
<p class="image-caption">Modular design for the upgraded smart rack.</p>

The new rack features 62 locks and looks professional. The cylindrical tubes you see aren't rocket launchers—they are custom compartments for the umbrellas, solving the tangling issue once and for all. This modular design not only expanded our capacity but also offered a novel, scalable approach to storing irregular items that traditional lockers simply can't handle.

<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/customlockers2.jpg"
    alt="Smart Rack V2 with 62 electric locks featuring custom cylindrical compartments to prevent umbrella tangling"
    style="margin-bottom: -15px"
/>
<img
    src="/assets/images/blogs/2025-12-08-automating-beach-rental-store/rackv2-cam.jpeg"
    alt="Smart Rack V2 with modular lock system, cylindrical umbrella tubes, and organized beach equipment storage visible from surveillance camera angle"
/>

<script src="https://player.vimeo.com/api/player.js"></script>
<p class="image-caption">Smart Rack V2 with modular locks and snug pipe compartments for umbrellas.</p>

## Conclusion

Automating **Beachin' Rentals** turned a struggling business into a profitable, self-running operation. By identifying the bottleneck (labor costs), embracing IoT technology, and iterating on hardware designs, we managed to create a unique rental experience that customers genuinely prefer.

The next steps? Scaling. We are looking to open more automated stores and explore taking this "smart rack" technology to other verticals that struggle with renting out irregularly shaped equipment.


[Thinking to automate your store?](/shop-automation/)

---

## Links

[Beachin' Rentals](https://beachinrentalsbcn.es/)
