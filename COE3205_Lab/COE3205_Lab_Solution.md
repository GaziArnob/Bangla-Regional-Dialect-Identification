# COE 3205 – Lab Task Solution
**8086 Assembly: I/O, Character Conversion, Register Operations**

## Part A: Code Writing

### Task 1 – Addition of two numbers (user input)

Reads two single-digit numbers (0–9) and prints their sum (0–18).

```asm
.MODEL SMALL
.STACK 100H
.DATA
    MSG1 DB 'Enter first number : $'
    MSG2 DB 0DH,0AH,'Enter second number: $'
    MSG3 DB 0DH,0AH,'Sum is : $'
.CODE
MAIN PROC
    MOV AX, @DATA
    MOV DS, AX

    LEA DX, MSG1        ; prompt for first number
    MOV AH, 09H
    INT 21H
    MOV AH, 01H         ; read digit -> AL (ASCII)
    INT 21H
    SUB AL, 30H         ; ASCII -> number
    MOV BL, AL          ; BL = first number

    LEA DX, MSG2        ; prompt for second number
    MOV AH, 09H
    INT 21H
    MOV AH, 01H
    INT 21H
    SUB AL, 30H         ; AL = second number

    ADD AL, BL          ; AL = sum (0..18)
    MOV AH, 0
    MOV BL, 10
    DIV BL              ; AL = tens digit, AH = units digit
    MOV BX, AX          ; save digits

    LEA DX, MSG3
    MOV AH, 09H
    INT 21H

    CMP BL, 0           ; skip tens digit if sum < 10
    JE  UNITS
    MOV DL, BL
    ADD DL, 30H         ; number -> ASCII
    MOV AH, 02H
    INT 21H
UNITS:
    MOV DL, BH          ; units digit
    ADD DL, 30H
    MOV AH, 02H
    INT 21H

    MOV AH, 4CH         ; exit
    INT 21H
MAIN ENDP
END MAIN
```
Sample: Input 4 and 3 → Output 7; Input 9 and 8 → Output 17

### Task 2 – Uppercase to Lowercase
(Given in the task sheet – `ADD AL, 20H`.) Sample: G → g

### Task 3 – Lowercase to Uppercase
(Given in the task sheet – `SUB AL, 20H`.) Sample: q → Q

### Task 4 – Hello World
(Given in the task sheet – `INT 21H / AH=09H` with `'Hello World$'`.)

---

## Part B: Register Trace Problems

All values in hexadecimal. 8-bit operations wrap around (e.g. FF+1 = 00, 00−1 = FF) and do not affect the other half of the register.

### Summary of answers

| # | AX | AH | AL | BX | BH | BL |
|---|---|---|---|---|---|---|
| 1 | 5982 | 59 | 82 | 1F83 | 1F | 83 |
| 2 | CD00 | CD | 00 | CD00 | CD | 00 |
| 3 | 8D7F | 8D | 7F | 8DFF | 8D | FF |
| 4 | 3D4D | 3D | 4D | 3C69 | 3C | 69 |
| 5 | 000C | 00 | 0C | 0CFF | 0C | FF |
| 6 | C3C2 | C3 | C2 | 19C2 | 19 | C2 |
| 7 | E8F0 | E8 | F0 | 932D | 93 | 2D |
| 8 | FE01 | FE | 01 | FE01 | FE | 01 |
| 9 | 7490 | 74 | 90 | 9168 | 91 | 68 |
| 10 | 5744 | 57 | 44 | 196E | 19 | 6E |
| 11 | A001 | A0 | 01 | FFA0 | FF | A0 |
| 12 | 6A4E | 6A | 4E | C4A5 | C4 | A5 |
| 13 | 00EF | 00 | EF | F000 | F0 | 00 |
| 14 | 5AA6 | 5A | A6 | 5A00 | 5A | 00 |
| 15 | 7520 | 75 | 20 | C8C8 | C8 | C8 |

### Step-by-step traces

### Problem 1

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,3A5C` | 3A5C | 0000 |
| `MOV BX,1F83` | 3A5C | 1F83 |
| `MOV AL,BL` | 3A83 | 1F83 |
| `ADD AH,BH` | 5983 | 1F83 |
| `DEC BL` | 5983 | 1F82 |
| `XCHG AL,BL` | 5982 | 1F83 |

**Final: AX = 5982H (AH=59, AL=82), BX = 1F83H (BH=1F, BL=83)**

### Problem 2

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,12FF` | 12FF | 0000 |
| `MOV BX,4501` | 12FF | 4501 |
| `INC AL` | 1200 | 4501 |
| `SUB AH,BH` | CD00 | 4501 |
| `MOV BH,AH` | CD00 | CD01 |
| `DEC BL` | CD00 | CD00 |

**Final: AX = CD00H (AH=CD, AL=00), BX = CD00H (BH=CD, BL=00)**

### Problem 3

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,8D20` | 8D20 | 0000 |
| `MOV BX,7FFE` | 8D20 | 7FFE |
| `INC BL` | 8D20 | 7FFF |
| `ADD AL,BL` | 8D1F | 7FFF |
| `MOV AL,AH` | 8D8D | 7FFF |
| `XCHG AL,BH` | 8D7F | 8DFF |

**Final: AX = 8D7FH (AH=8D, AL=7F), BX = 8DFFH (BH=8D, BL=FF)**

### Problem 4

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,A5B6` | A5B6 | 0000 |
| `MOV BX,3C4D` | A5B6 | 3C4D |
| `MOV AH,BH` | 3CB6 | 3C4D |
| `SUB AL,BL` | 3C69 | 3C4D |
| `INC BH` | 3C69 | 3D4D |
| `XCHG AX,BX` | 3D4D | 3C69 |

**Final: AX = 3D4DH (AH=3D, AL=4D), BX = 3C69H (BH=3C, BL=69)**

### Problem 5

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,010D` | 010D | 0000 |
| `MOV BX,00FF` | 010D | 00FF |
| `DEC AH` | 000D | 00FF |
| `DEC AL` | 000C | 00FF |
| `ADD BH,AL` | 000C | 0CFF |

**Final: AX = 000CH (AH=00, AL=0C), BX = 0CFFH (BH=0C, BL=FF)**

### Problem 6

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,56A9` | 56A9 | 0000 |
| `MOV BX,C3E7` | 56A9 | C3E7 |
| `SUB AL,BL` | 56C2 | C3E7 |
| `ADD AH,BH` | 19C2 | C3E7 |
| `MOV BL,AL` | 19C2 | C3C2 |
| `XCHG AH,BH` | C3C2 | 19C2 |

**Final: AX = C3C2H (AH=C3, AL=C2), BX = 19C2H (BH=19, BL=C2)**

### Problem 7

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,7B3D` | 7B3D | 0000 |
| `MOV BX,92F0` | 7B3D | 92F0 |
| `ADD AL,BL` | 7B2D | 92F0 |
| `INC BH` | 7B2D | 93F0 |
| `SUB AH,BH` | E82D | 93F0 |
| `XCHG AL,BL` | E8F0 | 932D |

**Final: AX = E8F0H (AH=E8, AL=F0), BX = 932DH (BH=93, BL=2D)**

### Problem 8

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,FFFF` | FFFF | 0000 |
| `MOV BX,0000` | FFFF | 0000 |
| `INC AL` | FF00 | 0000 |
| `INC BX` | FF00 | 0001 |
| `DEC AH` | FE00 | 0001 |
| `MOV AL,BL` | FE01 | 0001 |
| `ADD BH,AH` | FE01 | FE01 |

**Final: AX = FE01H (AH=FE, AL=01), BX = FE01H (BH=FE, BL=01)**

### Problem 9

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,4C91` | 4C91 | 0000 |
| `MOV BX,2BD8` | 4C91 | 2BD8 |
| `MOV BH,AL` | 4C91 | 91D8 |
| `SUB AH,BL` | 7491 | 91D8 |
| `DEC AL` | 7490 | 91D8 |
| `ADD BL,AL` | 7490 | 9168 |

**Final: AX = 7490H (AH=74, AL=90), BX = 9168H (BH=91, BL=68)**

### Problem 10

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,6E2A` | 6E2A | 0000 |
| `MOV BX,19C5` | 6E2A | 19C5 |
| `ADD AL,BH` | 6E43 | 19C5 |
| `SUB BL,AH` | 6E43 | 1957 |
| `XCHG AH,BL` | 5743 | 196E |
| `INC AL` | 5744 | 196E |

**Final: AX = 5744H (AH=57, AL=44), BX = 196EH (BH=19, BL=6E)**

### Problem 11

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,9F00` | 9F00 | 0000 |
| `MOV BX,01FF` | 9F00 | 01FF |
| `INC BL` | 9F00 | 0100 |
| `DEC AL` | 9FFF | 0100 |
| `ADD AH,BH` | A0FF | 0100 |
| `MOV BL,AH` | A0FF | 01A0 |
| `XCHG AL,BH` | A001 | FFA0 |

**Final: AX = A001H (AH=A0, AL=01), BX = FFA0H (BH=FF, BL=A0)**

### Problem 12

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,2D4E` | 2D4E | 0000 |
| `MOV BX,F1A6` | 2D4E | F1A6 |
| `SUB BH,AH` | 2D4E | C4A6 |
| `MOV AH,BL` | A64E | C4A6 |
| `ADD AH,BH` | 6A4E | C4A6 |
| `DEC BL` | 6A4E | C4A5 |

**Final: AX = 6A4EH (AH=6A, AL=4E), BX = C4A5H (BH=C4, BL=A5)**

### Problem 13

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,0F0F` | 0F0F | 0000 |
| `MOV BX,F0F0` | 0F0F | F0F0 |
| `ADD AX,BX` | FFFF | F0F0 |
| `INC AX` | 0000 | F0F0 |
| `DEC BX` | 0000 | F0EF |
| `XCHG AL,BL` | 00EF | F000 |

**Final: AX = 00EFH (AH=00, AL=EF), BX = F000H (BH=F0, BL=00)**

### Problem 14

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,5A5A` | 5A5A | 0000 |
| `MOV BX,A5A5` | 5A5A | A5A5 |
| `MOV AL,BH` | 5AA5 | A5A5 |
| `MOV BH,AH` | 5AA5 | 5AA5 |
| `SUB AL,BL` | 5A00 | 5AA5 |
| `INC BL` | 5A00 | 5AA6 |
| `XCHG AX,BX` | 5AA6 | 5A00 |

**Final: AX = 5AA6H (AH=5A, AL=A6), BX = 5A00H (BH=5A, BL=00)**

### Problem 15

| Instruction | AX | BX |
|---|---|---|
| `MOV AX,C8B7` | C8B7 | 0000 |
| `MOV BX,3E69` | C8B7 | 3E69 |
| `ADD AL,BL` | C820 | 3E69 |
| `SUB BH,AH` | C820 | 7669 |
| `DEC BH` | C820 | 7569 |
| `MOV BL,AH` | C820 | 75C8 |
| `XCHG AH,BH` | 7520 | C8C8 |

**Final: AX = 7520H (AH=75, AL=20), BX = C8C8H (BH=C8, BL=C8)**

